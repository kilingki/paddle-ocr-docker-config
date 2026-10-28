import asyncio
from dataclasses import replace
from io import BytesIO
import httpx
import pytest
from PIL import Image
from app.main import create_app
from app.config import Settings
from app.errors import APIError
from app.images import prepare_image
from app.schemas import parse_payload
from app.spotting import parse_spotting
from conftest import make_runtime


def png(width=64, height=32):
    stream = BytesIO()
    Image.new("RGB", (width, height), "white").save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("raw", ['{}','{"task":"unknown"}','{"task":"ocr","max_new_tokens":true}',
    '{"task":"ocr","max_new_tokens":1.2}','{"task":"ocr","max_new_tokens":"2"}',
    '{"task":"ocr","mode":"detect_read"}','{"task":"ocr","task":"table"}'])
def test_strict_payload(raw):
    with pytest.raises(APIError):
        parse_payload(raw, 4096)


def test_native_coordinate_parser_preserves_original_and_rejects_repairs():
    native = "text<|LOC_0|><|LOC_0|><|LOC_1000|><|LOC_0|><|LOC_1000|><|LOC_500|><|LOC_0|><|LOC_500|>"
    good = parse_spotting(native, 200, 100)
    assert good["regions"][0]["polygon_input_px"][2] == [200, 50]
    assert good["parse_status"] == "ok"
    bad = parse_spotting(native.replace("LOC_1000", "LOC_1001"), 200, 100)
    assert bad["parse_status"] == "invalid_or_incomplete" and bad["regions"] == []
    assert parse_spotting("", 100, 100)["parse_status"] != "ok"


def test_image_limits_and_no_exif_rotation():
    s = replace(Settings(), max_image_edge=64)
    assert prepare_image(png(), s).width == 64
    with pytest.raises(APIError):
        prepare_image(png(65, 1), s)
    with pytest.raises(APIError):
        prepare_image(b"not an image", s)


def test_http_contract_and_single_task_execution():
    async def scenario():
        rt = make_runtime()
        app = create_app(rt.settings, rt)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                assert (await client.get("/health")).json() == {"controller":"ok"}
                assert (await client.get("/control/status")).json()["state"] == "unloaded"
                assert (await client.post("/control/load", json={"model":"x"})).status_code == 400
                data = {"payload": '{"task":"ocr"}'}
                files = {"file": ("image.png", png(), "image/png")}
                assert (await client.post("/ocr", data=data, files=files)).status_code == 503
                assert (await client.post("/control/load", json={})).status_code == 200
                for task in ("spotting", "ocr", "formula", "table"):
                    result = await client.post("/ocr", data={"payload": '{"task":"'+task+'"}'}, files=files)
                    assert result.status_code == 200, result.text
                    body = result.json()
                    assert body["execution"]["sequences_started"] == 1
                    assert body["task"] == task
                assert len(rt.client.requests) == 4
                duplicate = [("file", ("a.png", png(), "image/png")), ("file", ("b.png", png(), "image/png"))]
                assert (await client.post("/ocr", data=data, files=duplicate)).status_code == 400
                assert (await client.post("/control/unload", content="null")).status_code == 400
                assert (await client.post("/control/unload", json={})).status_code == 200
    asyncio.run(scenario())


def test_chunked_body_is_limited_before_parsing():
    async def scenario():
        rt = make_runtime(max_body_bytes=64)
        app = create_app(rt.settings, rt)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                async def chunks():
                    yield b"x" * 40
                    yield b"y" * 40
                response = await client.post("/ocr", content=chunks(), headers={"content-type":"multipart/form-data; boundary=x"})
                assert response.status_code == 413
                assert not rt.client.requests
    asyncio.run(scenario())


def test_oversized_control_body_is_bad_request_without_loading():
    async def scenario():
        rt=make_runtime()
        app=create_app(rt.settings,rt)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test") as client:
                response=await client.post("/control/load",content=b"x"*2048)
                assert response.status_code==400
                assert response.json()["error"]["code"]=="BAD_REQUEST"
                assert rt.worker.starts==0
    asyncio.run(scenario())
