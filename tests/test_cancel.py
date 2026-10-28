import asyncio
import pytest
from app.errors import APIError
from app.schemas import Payload
from conftest import image, make_runtime, until


def test_atomic_slot_busy_and_uncertain_cancel_preserves_active():
    async def scenario():
        rt = make_runtime()
        rt.client.gate.clear()
        rt.client.idle_value = False
        await rt.control("load")
        ticket = await rt.submit(image(), Payload(task="ocr"))
        async def started():
            return bool(rt.client.requests)
        await until(started)
        assert (await rt.control("load"))["active_requests"] == 1
        with pytest.raises(APIError) as exc:
            await rt.submit(image(), Payload(task="ocr"))
        assert exc.value.status == 429
        with pytest.raises(APIError) as exc:
            await rt.control("unload")
        assert exc.value.code == "BUSY"
        await rt.cancel(ticket)
        assert (await rt.status())["active_requests"] == 1
        with pytest.raises(APIError):
            await rt.control("unload")
        rt.client.idle_value = True
        async def cleared():
            return (await rt.status())["active_requests"] == 0
        await until(cleared)
        await rt._release(ticket)
        assert (await rt.status())["active_requests"] == 0
        assert rt.worker.live
        await rt.close()
    asyncio.run(scenario())


def test_cancel_one_does_not_kill_another_and_stops_new_admission():
    async def scenario():
        rt = make_runtime(max_active=2)
        rt.client.gate.clear()
        await rt.control("load")
        one = await rt.submit(image(), Payload(task="ocr"))
        two = await rt.submit(image(), Payload(task="table"))
        async def started():
            return len(rt.client.requests) == 2
        await until(started)
        await rt.cancel(one)
        assert not two.task.done()
        with pytest.raises(APIError) as exc:
            await rt.submit(image(), Payload(task="formula"))
        assert exc.value.status == 429
        rt.client.gate.set()
        result = await two.task
        assert result["task"] == "table"
        async def cleared():
            return (await rt.status())["active_requests"] == 0
        await until(cleared)
        assert rt.worker.live and rt.worker.stops == 0
        await rt.close()
    asyncio.run(scenario())


def test_cancel_before_coroutine_start_never_leaks_slot():
    async def scenario():
        rt = make_runtime()
        await rt.control("load")
        ticket = await rt.submit(image(), Payload(task="ocr"))
        await rt.cancel(ticket)
        async def cleared():
            return (await rt.status())["active_requests"] == 0
        await until(cleared)
        assert len(rt.client.requests) == 0
        await rt.close()
    asyncio.run(scenario())


def test_request_timeout_remains_busy_until_engine_idle():
    async def scenario():
        rt = make_runtime(request_timeout=.03)
        rt.client.gate.clear()
        rt.client.idle_value = False
        await rt.control("load")
        ticket = await rt.submit(image(), Payload(task="ocr"))
        result = await ticket.task
        assert result.status == 504
        assert (await rt.status())["active_requests"] == 1
        rt.client.idle_value = True
        async def cleared():
            return (await rt.status())["active_requests"] == 0
        await until(cleared)
        await rt.close()
    asyncio.run(scenario())


def test_unload_first_prevents_inference_admission():
    async def scenario():
        rt = make_runtime()
        await rt.control("load")
        rt.worker.stop_gate.clear()
        unloading = asyncio.create_task(rt.control("unload"))
        async def started():
            return (await rt.status())["state"] == "unloading"
        await until(started)
        with pytest.raises(APIError) as exc:
            await rt.submit(image(), Payload(task="ocr"))
        assert exc.value.status == 503
        rt.worker.stop_gate.set()
        await unloading
        assert not rt.client.requests
        await rt.close()
    asyncio.run(scenario())
