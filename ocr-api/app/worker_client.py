import json, re
import httpx
from .config import PROMPTS


class WorkerClient:
    def __init__(self, settings):
        self.settings = settings
        self.partials = {}
        self.http = httpx.AsyncClient(
            base_url=settings.worker_url, trust_env=False,
            timeout=httpx.Timeout(5, read=None),
            limits=httpx.Limits(max_connections=settings.max_active + 8, max_keepalive_connections=8))

    def take_partial(self, request_id):
        return self.partials.pop(request_id, None)

    async def close(self):
        await self.http.aclose()

    async def models(self):
        response = await self.http.get("/v1/models", timeout=2)
        response.raise_for_status()
        return response.json()

    async def healthy(self):
        response = await self.http.get("/health", timeout=2)
        response.raise_for_status()

    async def idle(self):
        response = await self.http.get("/metrics", timeout=2)
        response.raise_for_status()
        values = []
        for name in ("num_requests_running", "num_requests_waiting"):
            matches = re.findall(r"^vllm:" + name + r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)", response.text, re.M)
            if not matches:
                raise RuntimeError("private engine activity metrics unavailable")
            values.append(sum(float(x) for x in matches))
        return values == [0, 0]

    async def generate(self, request_id, image, payload):
        s = self.settings
        body = {"model": s.model_id, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image.data_url}},
            {"type": "text", "text": PROMPTS[payload.task]}]}],
            "temperature": 0, "top_p": 1, "n": 1, "max_tokens": payload.max_new_tokens,
            "skip_special_tokens": False, "spaces_between_special_tokens": False,
            "stream": True, "stream_options": {"include_usage": True},
            "mm_processor_kwargs": {"min_pixels": s.min_pixels,
                "max_pixels": s.spotting_max_pixels if payload.task == "spotting" else s.reading_max_pixels}}
        parts, usage, finish, stop, done, received = [], None, None, None, False, 0
        # Closing this context on cancellation propagates disconnect to the private vLLM route.
        try:
            async with self.http.stream("POST", "/v1/chat/completions", json=body,
                                        headers={"X-Request-Id": request_id}) as response:
                if response.status_code != 200:
                    raw = await response.aread()
                    raise RuntimeError(f"private inference HTTP {response.status_code}: {raw[:512]!r}")
                async for line in response.aiter_lines():
                    received += len(line)
                    if received > 8 * 1024 * 1024:
                        raise RuntimeError("private output exceeds response limit")
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        done = True
                        break
                    event = json.loads(line[6:])
                    if "error" in event:
                        raise RuntimeError("private inference stream error")
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        if choice["index"] != 0:
                            raise RuntimeError("unexpected multiple sequences")
                        content = choice.get("delta", {}).get("content")
                        if content:
                            parts.append(content)
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
                            stop = choice.get("stop_reason")
            if not done or finish is None or usage is None:
                raise RuntimeError("private stream ended without terminal/usage")
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if type(usage.get(key)) is not int or usage[key] < 0:
                    raise RuntimeError("invalid private usage")
            return {"native_text": "".join(parts), "finish_reason": finish,
                    "stop_reason": stop, "usage": usage}
        except BaseException:
            if parts:
                self.partials[request_id] = {"native_text": "".join(parts), "usage": usage,
                                             "finish_reason": finish, "technical_status": "incomplete"}
            raise
