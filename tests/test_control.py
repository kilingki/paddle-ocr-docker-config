import asyncio
import pytest
from app.errors import APIError
from app.schemas import control_body
from conftest import make_runtime, until


@pytest.mark.parametrize("body", [b"{}", b"", b"  { } "])
def test_control_empty_body(body):
    control_body(body)


@pytest.mark.parametrize("body", [b"null", b"[]", b'{"model":"x"}', b'{"a":1,"a":2}', b"{", b"false"])
def test_control_rejects_other_bodies(body):
    with pytest.raises(APIError):
        control_body(body)


def test_shared_load_survives_cancelled_waiter_and_conflicts():
    async def scenario():
        rt = make_runtime()
        rt.worker.start_gate.clear()
        first = asyncio.create_task(rt.control("load"))
        async def loading():
            return (await rt.status())["state"] == "loading"
        await until(loading)
        second = asyncio.create_task(rt.control("load"))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(APIError) as exc:
            await rt.control("unload")
        assert exc.value.code == "LIFECYCLE_CONFLICT"
        assert (await rt.status())["active_requests"] == 0
        rt.worker.start_gate.set()
        result = await second
        assert result == {"state": "ready", "residency": "resident", "active_requests": 0, "last_error": None}
        assert rt.worker.starts == 1
        await rt.control("load")
        assert rt.worker.starts == 1
        await rt.control("unload")
        assert (await rt.status())["residency"] == "not_resident"
        await rt.close()
    asyncio.run(scenario())


def test_duplicate_unload_shares_snapshot_and_blocks_new_load():
    async def scenario():
        rt = make_runtime()
        await rt.control("load")
        rt.worker.stop_gate.clear()
        one = asyncio.create_task(rt.control("unload"))
        async def unloading():
            return (await rt.status())["state"] == "unloading"
        await until(unloading)
        two = asyncio.create_task(rt.control("unload"))
        with pytest.raises(APIError) as exc:
            await rt.control("load")
        assert exc.value.code == "LIFECYCLE_CONFLICT"
        rt.worker.stop_gate.set()
        a, b = await asyncio.gather(one, two)
        assert a == b and a["state"] == "unloaded"
        assert rt.worker.stops == 1
        await rt.close()
    asyncio.run(scenario())


def test_failed_load_and_failed_unload_require_confirmed_cleanup():
    async def scenario():
        rt = make_runtime()
        rt.worker.fail_start = True
        with pytest.raises(APIError):
            await rt.control("load")
        assert (await rt.status())["residency"] == "not_resident"
        rt.worker.fail_start = False
        await rt.control("load")
        rt.worker.fail_stop = True
        with pytest.raises(APIError):
            await rt.control("unload")
        status = await rt.status()
        assert status["state"] == "failed" and status["residency"] == "unknown"
        with pytest.raises(APIError):
            await rt.control("load")
        rt.worker.fail_stop = False
        await rt.control("unload")
        assert (await rt.status())["last_error"] is None
        await rt.close()
    asyncio.run(scenario())


def test_failed_cleanup_later_converges_without_reloading():
    async def scenario():
        rt = make_runtime()
        await rt.control("load")
        rt.worker.fail_stop = True
        with pytest.raises(APIError):
            await rt.control("unload")
        assert (await rt.status())["residency"] == "unknown"
        rt.worker.fail_stop = False
        async def recovered():
            return (await rt.status())["residency"] == "not_resident"
        await until(recovered)
        assert rt.worker.starts == 1
        assert (await rt.status())["state"] == "failed"
        await rt.control("unload")
        assert (await rt.status())["state"] == "unloaded"
        await rt.close()
    asyncio.run(scenario())

