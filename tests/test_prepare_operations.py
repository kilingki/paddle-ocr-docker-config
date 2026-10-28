from io import BytesIO
import json
import pytest
from app import prepare as module


def fake_commands(calls, running):
    def command(root, args):
        calls.append(args)
        if args[1:4] == ["compose","config","--format"]:
            return json.dumps({"name":"test","services":{"paddle-ocr-runtime":{
                "image":"paddle-ocr-runtime:local","ports":[{"target":8000,"published":"8020","host_ip":"127.0.0.1"}]}}})
        if "ps" in args:
            return "container123"
        if args[1] == "inspect":
            return json.dumps([{"Config":{"Labels":{"com.docker.compose.project":"test","com.docker.compose.service":"paddle-ocr-runtime","com.docker.compose.project.working_dir":str(root)}},
                "HostConfig":{"PortBindings":{"8000/tcp":[{"HostIp":"127.0.0.1","HostPort":"8020"}]}},
                "State":{"Running":running}}])
        raise AssertionError("unexpected mutating command: " + str(args))
    return command


def test_prepare_preserves_existing_active_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module, "command", fake_commands(calls, True))
    value = {"state":"ready","residency":"resident","active_requests":2,"last_error":None}
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a,**kw:BytesIO(json.dumps(value).encode()))
    assert module.prepare(tmp_path) == value
    assert not any("up" in args or "restart" in args for args in calls)


def test_prepare_status_failure_does_not_restart(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module, "command", fake_commands(calls, True))
    ticks = iter([0,0,20])
    monkeypatch.setattr(module.time, "monotonic", lambda:next(ticks))
    monkeypatch.setattr(module.time, "sleep", lambda value:None)
    def unavailable(*a,**kw):
        raise OSError("endpoint timeout")
    monkeypatch.setattr(module.urllib.request, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="not restarted"):
        module.prepare(tmp_path)
    assert not any("up" in args or "restart" in args for args in calls)


def test_prepare_rejects_same_service_from_another_worktree(tmp_path, monkeypatch):
    calls=[]
    original=fake_commands(calls,True)
    def command(root,args):
        result=original(root,args)
        if args[1]=="inspect":
            payload=json.loads(result)
            payload[0]["Config"]["Labels"]["com.docker.compose.project.working_dir"]="/different/worktree"
            return json.dumps(payload)
        return result
    monkeypatch.setattr(module,"command",command)
    with pytest.raises(RuntimeError,match="identity mismatch"):
        module.prepare(tmp_path)
    assert not any("up" in args or "restart" in args for args in calls)



def test_stopped_container_is_started_by_compose_and_returns_unloaded(tmp_path, monkeypatch):
    calls=[]
    original=fake_commands(calls,False)
    def command(root,args):
        if args[1:3]==["image","inspect"]:
            calls.append(args)
            return "[]"
        if "up" in args:
            calls.append(args)
            return "started"
        return original(root,args)
    monkeypatch.setattr(module,"command",command)
    value={"state":"unloaded","residency":"not_resident","active_requests":0,"last_error":None}
    monkeypatch.setattr(module.urllib.request,"urlopen",lambda *a,**kw:BytesIO(json.dumps(value).encode()))
    assert module.prepare(tmp_path)==value
    assert any("up" in args and "--no-build" in args and "--no-recreate" in args for args in calls)

