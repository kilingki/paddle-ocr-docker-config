import asyncio
import httpx
import pytest
from app.config import Settings
from app.schemas import Payload
from app.worker_client import WorkerClient
from conftest import image


class BrokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"index":0,"delta":{"content":"raw-prefix"},"finish_reason":null}]}\n\n'
        raise httpx.ReadError("stream broke")


def test_partial_native_output_survives_upstream_failure():
    async def scenario():
        client=WorkerClient(Settings())
        await client.http.aclose()
        client.http=httpx.AsyncClient(base_url="http://private",transport=httpx.MockTransport(
            lambda request:httpx.Response(200,stream=BrokenStream())))
        with pytest.raises(httpx.ReadError):
            await client.generate("request",image(),Payload(task="ocr"))
        partial=client.take_partial("request")
        assert partial["native_text"]=="raw-prefix"
        assert partial["technical_status"]=="incomplete"
        assert client.take_partial("request") is None
        await client.close()
    asyncio.run(scenario())


def test_fatal_error_retains_and_releases_partial_response():
    from app.schemas import Payload
    from conftest import image, make_runtime, until
    async def scenario():
        rt=make_runtime()
        rt.client.gate.clear()
        await rt.control("load")
        ticket=await rt.submit(image(),Payload(task="ocr"))
        async def started():
            return bool(rt.client.requests)
        await until(started)
        partials={ticket.request_id:{"native_text":"partial-before-fatal","technical_status":"incomplete"}}
        rt.client.take_partial=lambda request_id:partials.pop(request_id,None)
        await rt._fatal(rt.generation,"engine crash")
        result=await ticket.task
        assert result.status==500
        assert result.body()["error"]["partial_result"]["native_text"]=="partial-before-fatal"
        assert partials=={}
        await rt.close()
    asyncio.run(scenario())

