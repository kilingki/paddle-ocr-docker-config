import asyncio
from dataclasses import replace
from app.config import Settings
from app.lifecycle import Runtime
from app.images import PreparedImage


class FakeWorker:
    def __init__(self):
        self.live = False
        self.starts = self.stops = 0
        self.start_gate = asyncio.Event()
        self.start_gate.set()
        self.stop_gate = asyncio.Event()
        self.stop_gate.set()
        self.fail_start = False
        self.fail_stop = False

    async def start(self, generation):
        self.starts += 1
        self.live = True
        await self.start_gate.wait()
        if self.fail_start:
            raise RuntimeError("fake load failure")

    async def stop(self):
        self.stops += 1
        await self.stop_gate.wait()
        if self.fail_stop:
            raise RuntimeError("fake stop failure")
        self.live = False

    def alive(self):
        return self.live

    def snapshot_children(self):
        pass


class FakeClient:
    def __init__(self):
        self.requests = {}
        self.gate = asyncio.Event()
        self.gate.set()
        self.idle_value = True

    async def generate(self, request_id, image, payload):
        self.requests[request_id] = payload.task
        await self.gate.wait()
        text = ("hello<|LOC_0|><|LOC_0|><|LOC_500|><|LOC_0|>"
                "<|LOC_500|><|LOC_500|><|LOC_0|><|LOC_500|>") if payload.task == "spotting" else "native text"
        return {"native_text": text, "finish_reason": "stop", "stop_reason": 2,
                "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}}

    async def healthy(self):
        pass

    async def idle(self):
        return self.idle_value

    async def close(self):
        pass


def make_runtime(**values):
    settings = replace(Settings(), health_interval=60, **values)
    client, worker = FakeClient(), FakeWorker()
    return Runtime(settings, worker, client)


def image():
    return PreparedImage("hash", 100, 50, "PNG", "data:image/png;base64,")


async def until(predicate, timeout=2):
    async with asyncio.timeout(timeout):
        while not await predicate():
            await asyncio.sleep(.01)
