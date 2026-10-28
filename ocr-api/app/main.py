import asyncio, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from .config import Settings
from .errors import APIError
from .images import prepare_image
from .schemas import parse_payload, control_body
from .engine import Worker
from .worker_client import WorkerClient
from .lifecycle import Runtime


class BodyLimitMiddleware:
    """Bound ingress before multipart parsing, including chunked bodies."""
    def __init__(self, app, settings):
        self.app, self.settings = app, settings
        self.inflight = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        ocr = scope["path"] == "/ocr"
        if ocr and self.inflight >= self.settings.max_inflight_uploads:
            return await JSONResponse(APIError(429, "TOO_MANY_REQUESTS", "upload slots occupied").body(), 429)(scope, receive, send)
        limit = self.settings.max_body_bytes if ocr else 1024
        overflow = (APIError(413, "BODY_TOO_LARGE", "request body exceeds limit") if ocr
                    else APIError(400, "BAD_REQUEST", "control body must be empty or {}"))
        headers = [v for k, v in scope["headers"] if k == b"content-length"]
        if len(headers) > 1 or (headers and (not headers[0].isdigit())):
            return await JSONResponse(APIError(400, "BAD_REQUEST", "invalid Content-Length").body(), 400)(scope, receive, send)
        if headers and int(headers[0]) > limit:
            return await JSONResponse(overflow.body(), overflow.status)(scope, receive, send)
        if ocr:
            self.inflight += 1
        try:
            chunks, size = [], 0
            try:
                async with asyncio.timeout(self.settings.upload_timeout):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        chunk = message.get("body", b"")
                        size += len(chunk)
                        if size > limit:
                            return await JSONResponse(overflow.body(), overflow.status)(scope, receive, send)
                        chunks.append(chunk)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                return await JSONResponse(APIError(408, "REQUEST_TIMEOUT", "upload deadline exceeded").body(), 408)(scope, receive, send)
            body, delivered = b"".join(chunks), False

            async def buffered_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, buffered_receive, send)
        finally:
            if ocr:
                self.inflight -= 1


async def disconnected(request):
    while True:
        event = await request.receive()
        if event["type"] == "http.disconnect":
            return


def create_app(settings=None, runtime=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        client = None
        if runtime is None:
            client = WorkerClient(settings)
            app.state.runtime = Runtime(settings, Worker(settings, client), client)
        else:
            app.state.runtime = runtime
        app.state.decode_slots = asyncio.Semaphore(settings.max_inflight_uploads)
        try:
            yield
        finally:
            await app.state.runtime.close()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, settings=settings)

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(APIError(400, "BAD_REQUEST", "invalid request").body(), 400)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        code = "BAD_REQUEST" if exc.status_code in (400, 422) else "NOT_FOUND"
        status = 400 if exc.status_code == 422 else exc.status_code
        return JSONResponse(APIError(status, code, str(exc.detail)).body(), status)

    @app.get("/health")
    async def health():
        return {"controller": "ok"}

    @app.get("/control/status")
    async def status(request: Request):
        try:
            return await request.app.state.runtime.status()
        except Exception as exc:
            raise APIError(500, "STATUS_FAILED", "cannot create trustworthy status") from exc

    @app.post("/control/{operation}")
    async def control(operation: str, request: Request):
        if operation not in ("load", "unload"):
            raise APIError(404, "NOT_FOUND", "unsupported control operation")
        control_body(await request.body())
        return await request.app.state.runtime.control(operation)

    @app.post("/ocr")
    async def ocr(request: Request):
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            raise APIError(400, "BAD_REQUEST", "expected multipart file and payload")
        async with request.form(max_files=1, max_fields=1, max_part_size=settings.max_body_bytes) as form:
            pairs = form.multi_items()
            if len(pairs) != 2 or set(form.keys()) != {"file", "payload"}:
                raise APIError(400, "BAD_REQUEST", "expected exactly one file and one payload")
            upload, raw = form["file"], form["payload"]
            if not isinstance(upload, UploadFile) or not isinstance(raw, str) or len(raw) > 4096:
                raise APIError(400, "BAD_REQUEST", "invalid multipart fields")
            payload = parse_payload(raw, settings.max_new_tokens)
            data = await upload.read()
        async with request.app.state.decode_slots:
            decode = asyncio.create_task(asyncio.to_thread(prepare_image, data, settings))
            try:
                image = await asyncio.shield(decode)
            except asyncio.CancelledError:
                # Bound CPU decoding even when its HTTP waiter disappears.
                await asyncio.shield(decode)
                raise
        if await request.is_disconnected():
            return Response(status_code=499)
        rt = request.app.state.runtime
        ticket = await rt.submit(image, payload)
        watcher = asyncio.create_task(disconnected(request))
        try:
            done, _ = await asyncio.wait((ticket.task, watcher), return_when=asyncio.FIRST_COMPLETED)
            if ticket.task not in done:
                await rt.cancel(ticket)
                return Response(status_code=499)
            result = ticket.task.result()
            if isinstance(result, APIError):
                return JSONResponse(result.body(request_id=ticket.request_id, stage="inference", task=payload.task, input=image.metadata()), result.status)
            return result
        except asyncio.CancelledError:
            await rt.cancel(ticket)
            raise
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    return app


logging.basicConfig(level=logging.INFO)
app = create_app()
