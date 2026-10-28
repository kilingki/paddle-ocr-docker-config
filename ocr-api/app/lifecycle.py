import asyncio, logging, time, uuid
from dataclasses import dataclass
from .errors import APIError
from .spotting import display_text, parse_spotting

log = logging.getLogger("paddle.lifecycle")


@dataclass
class Ticket:
    request_id: str
    generation: int
    task: asyncio.Task | None = None
    pending: bool = False
    started: bool = False
    cancel_requested: bool = False
    abort_error: APIError | None = None


class Runtime:
    def __init__(self, settings, worker, client):
        self.settings, self.worker, self.client = settings, worker, client
        self.lock = asyncio.Lock()
        self.state, self.residency, self.last_error = "unloaded", "not_resident", None
        self.generation = 0
        self.tickets = {}
        self.operation = None
        self.operation_kind = None
        self.recovery_task = None
        self.watch_task = None
        self.fatal_task = None
        self.cleanup_task = None
        self.draining = False
        self.closing = False

    def _status(self):
        return {"state": self.state, "residency": self.residency,
                "active_requests": len(self.tickets), "last_error": self.last_error}

    async def status(self):
        async with self.lock:
            return self._status()

    async def control(self, kind):
        async with self.lock:
            active = len(self.tickets)
            if kind == "unload" and self.operation_kind == "unload" and self.operation and not self.operation.done():
                operation = self.operation
            elif kind == "unload" and active:
                raise APIError(409, "BUSY", "inference or cancellation cleanup is active")
            elif self.operation and not self.operation.done():
                if self.operation_kind != kind:
                    raise APIError(409, "LIFECYCLE_CONFLICT", "opposite lifecycle operation is in progress")
                operation = self.operation
            elif kind == "load" and self.state == "ready":
                return self._status()
            elif kind == "unload" and self.residency == "not_resident":
                self.state, self.last_error = "unloaded", None
                return self._status()
            elif kind == "load" and (self.residency != "not_resident" or active):
                raise APIError(409, "LIFECYCLE_CONFLICT", "residency must be cleared before load")
            else:
                if self.closing:
                    raise APIError(409, "LIFECYCLE_CONFLICT", "controller is shutting down")
                self.state = "loading" if kind == "load" else "unloading"
                if kind == "load":
                    self.generation += 1
                    self.residency = "unknown"
                self.operation_kind = kind
                operation = self.operation = asyncio.create_task(self._operate(kind), name=f"control-{kind}")
        # The operation outlives the HTTP waiter and owns its immutable final snapshot.
        return await asyncio.shield(operation)

    async def _operate(self, kind):
        try:
            if kind == "load":
                async with asyncio.timeout(self.settings.load_timeout):
                    await self.worker.start(self.generation)
                async with self.lock:
                    self.state, self.residency, self.last_error = "ready", "resident", None
                    result = self._status()
                if self.watch_task:
                    self.watch_task.cancel()
                self.watch_task = asyncio.create_task(self._watch(self.generation), name="worker-watch")
                return result
            await self.worker.stop()
            async with self.lock:
                self.state, self.residency, self.last_error = "unloaded", "not_resident", None
                return self._status()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            cleared = False
            if kind == "load":
                try:
                    await self.worker.stop()
                    cleared = True
                except Exception:
                    log.exception("worker cleanup failed")
            code = "LOAD_FAILED" if kind == "load" else "UNLOAD_FAILED"
            log.exception("%s", code)
            async with self.lock:
                self.state = "failed"
                self.residency = "not_resident" if cleared else "unknown"
                self.last_error = {"code": code, "message": (type(exc).__name__ + ": " + str(exc))[:1000]}
            if not cleared:
                self._schedule_failed_cleanup(self.generation)
            raise APIError(500, code, (type(exc).__name__ + ": " + str(exc))[:1000]) from exc

    async def submit(self, image, payload):
        async with self.lock:
            if self.state != "ready" or self.closing:
                raise APIError(503, "NOT_READY", "model is not ready")
            if self.draining or len(self.tickets) >= self.settings.max_active:
                raise APIError(429, "TOO_MANY_REQUESTS", "execution slots are occupied")
            ticket = Ticket(uuid.uuid4().hex, self.generation)
            self.tickets[ticket.request_id] = ticket
            ticket.task = asyncio.create_task(self._execute(ticket, image, payload),
                                               name=f"ocr-{ticket.request_id}")
            return ticket

    async def _release(self, ticket):
        async with self.lock:
            if ticket.generation == self.generation:
                self.tickets.pop(ticket.request_id, None)

    async def _unconfirmed(self, ticket):
        async with self.lock:
            if ticket.request_id not in self.tickets or ticket.generation != self.generation:
                return
            ticket.pending = True
            self.draining = True
            if self.recovery_task is None or self.recovery_task.done():
                self.recovery_task = asyncio.create_task(self._recover(ticket.generation), name="cancel-drain")

    def _request_error(self, ticket, status, code, message):
        error = APIError(status, code, message)
        getter = getattr(self.client, "take_partial", None)
        partial = getter(ticket.request_id) if getter else None
        if partial is not None:
            error.details["partial_result"] = partial
        return error

    async def _execute(self, ticket, image, payload):
        ticket.started = True
        if ticket.cancel_requested:
            await self._release(ticket)
            return ticket.abort_error or APIError(499, "REQUEST_CANCELLED", "cancelled before engine admission")
        try:
            async with asyncio.timeout(self.settings.request_timeout):
                result = await self.client.generate(ticket.request_id, image, payload)
            native = result["native_text"]
            complete = result["finish_reason"] == "stop" and bool(native)
            output = {"request_id": ticket.request_id, "model": self.settings.model_id,
                      "revision": self.settings.revision, "task": payload.task,
                      "input": image.metadata(), **result, "text": display_text(native),
                      "technical_status": "ok" if complete else "incomplete",
                      "execution": {"sequences_started": 1, "automatic_retries": 0}}
            if payload.task == "spotting":
                output.update(parse_spotting(native, image.width, image.height))
                if output["parse_status"] != "ok":
                    output["technical_status"] = "incomplete"
            await self._release(ticket)
            return output
        except asyncio.CancelledError:
            await self._unconfirmed(ticket)
            if ticket.abort_error:
                return self._request_error(ticket, ticket.abort_error.status, ticket.abort_error.code, ticket.abort_error.message)
            return self._request_error(ticket, 499, "REQUEST_CANCELLED", "client disconnected")
        except TimeoutError:
            await self._unconfirmed(ticket)
            return self._request_error(ticket, 504, "INFERENCE_TIMEOUT", "inference deadline exceeded; cancellation cleanup may remain active")
        except Exception as exc:
            log.exception("inference failed request=%s", ticket.request_id)
            await self._unconfirmed(ticket)
            return self._request_error(ticket, 500, "INFERENCE_FAILED", (type(exc).__name__ + ": " + str(exc))[:1000])

    async def cancel(self, ticket):
        ticket.cancel_requested = True
        if ticket.task and ticket.started and not ticket.task.done():
            ticket.task.cancel()
        if ticket.task:
            # Cancellation does not abandon the task's cleanup.
            await asyncio.shield(ticket.task)

    async def _recover(self, generation):
        idle_observations = 0
        cleanup_start = None
        while not self.closing:
            async with self.lock:
                if generation != self.generation:
                    return
                pending = [t for t in self.tickets.values() if t.pending]
                running = [t for t in self.tickets.values() if not t.pending]
                if not pending:
                    self.draining = False
                    return
            # Do not kill other healthy requests just because an aggregate ACK is delayed.
            if running:
                cleanup_start = None
                await asyncio.sleep(.1)
                continue
            cleanup_start = cleanup_start or time.monotonic()
            try:
                await self.client.healthy()
                idle = self.worker.alive() and await self.client.idle()
                idle_observations = idle_observations + 1 if idle else 0
                if idle_observations >= 2:
                    async with self.lock:
                        if generation == self.generation:
                            for ticket in pending:
                                self.tickets.pop(ticket.request_id, None)
                            self.draining = False
                    return
            except Exception:
                idle_observations = 0
            if time.monotonic() - cleanup_start >= self.settings.cancel_timeout:
                await self._fatal(generation, "cancellation terminal could not be confirmed")
                return
            await asyncio.sleep(.1)

    async def _watch(self, generation):
        failures, next_health = 0, time.monotonic() + self.settings.health_interval
        while not self.closing:
            await asyncio.sleep(.2)
            async with self.lock:
                if self.state != "ready" or generation != self.generation:
                    return
            self.worker.snapshot_children()
            failed = not self.worker.alive()
            if time.monotonic() >= next_health:
                next_health = time.monotonic() + self.settings.health_interval
                try:
                    await self.client.healthy()
                    failures = 0
                except Exception:
                    failures += 1
            if failed or failures >= 3:
                await self._fatal(generation, "owned vLLM engine became unavailable")
                return

    async def _fatal(self, generation, message):
        async with self.lock:
            if generation != self.generation or self.state != "ready":
                return
            if self.fatal_task and self.fatal_task is not asyncio.current_task() and not self.fatal_task.done():
                return
            self.fatal_task = asyncio.current_task()
            self.state, self.residency = "failed", "unknown"
            self.last_error = {"code": "INFERENCE_FAILED", "message": message}
            tickets = list(self.tickets.values())
        for ticket in tickets:
            ticket.cancel_requested = True
            ticket.abort_error = APIError(500, "INFERENCE_FAILED", message)
            if ticket.task and ticket.started and ticket.task is not asyncio.current_task() and not ticket.task.done():
                ticket.task.cancel()
        try:
            await self.worker.stop()
        except Exception:
            log.exception("fatal cleanup could not prove termination")
            self._schedule_failed_cleanup(generation)
            return
        async with self.lock:
            if generation == self.generation:
                self.residency = "not_resident"
                self.tickets.clear()
                self.draining = False

    def _schedule_failed_cleanup(self, generation):
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._retry_failed_cleanup(generation), name="failed-worker-cleanup")

    async def _retry_failed_cleanup(self, generation):
        # Preserve unknown/active until the owned process tree is actually gone.
        # This never reloads a model or signals an unrelated process.
        while not self.closing:
            await asyncio.sleep(.5)
            async with self.lock:
                if generation != self.generation or self.state != "failed" or self.residency != "unknown":
                    return
            try:
                await self.worker.stop()
            except Exception:
                continue
            async with self.lock:
                if generation == self.generation and self.state == "failed":
                    self.residency = "not_resident"
                    self.tickets.clear()
                    self.draining = False
            return

    async def close(self):
        self.closing = True
        background = [t for t in (self.watch_task, self.recovery_task, self.cleanup_task, self.operation) if t]
        for task in background:
            if task and not task.done():
                task.cancel()
        tasks = [t.task for t in self.tickets.values() if t.task and not t.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*background, return_exceptions=True)
        await self.worker.stop()
        await self.client.close()
