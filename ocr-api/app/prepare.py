"""Host-only preparation; standard library, no model/framework imports."""
from pathlib import Path
import fcntl, json, os, subprocess, sys, time, urllib.request

SERVICE = "paddle-ocr-runtime"


def status_valid(value):
    if not isinstance(value, dict) or set(value) != {"state", "residency", "active_requests", "last_error"}:
        return False
    state, residency, active, error = (value[k] for k in ("state", "residency", "active_requests", "last_error"))
    if state not in ("unloaded", "loading", "ready", "unloading", "failed") or residency not in ("resident", "not_resident", "unknown"):
        return False
    if type(active) is not int or active < 0:
        return False
    if error is not None and (not isinstance(error, dict) or set(error) != {"code", "message"}
                              or not all(isinstance(error[k], str) for k in error) or not error["code"]):
        return False
    return not ((state == "ready" and residency != "resident")
                or (state == "unloaded" and (residency != "not_resident" or active != 0)))


def command(root, args):
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def prepare(root):
    root = Path(root).resolve()
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cfg = json.loads(command(root, ["docker", "compose", "config", "--format", "json"]))
        service = cfg["services"][SERVICE]
        ports = [p for p in service["ports"] if p["target"] == 8000]
        if len(ports) != 1 or ports[0].get("host_ip") != "127.0.0.1":
            raise RuntimeError("expected exactly one loopback public port")
        port = int(ports[0]["published"])
        ids = command(root, ["docker", "compose", "ps", "--all", "--quiet", SERVICE]).split()
        if len(ids) > 1:
            raise RuntimeError("ambiguous Compose instance")
        existing = None
        if ids:
            existing = json.loads(command(root, ["docker", "inspect", ids[0]]))[0]
            labels = existing["Config"].get("Labels", {})
            if (labels.get("com.docker.compose.project") != cfg["name"]
                    or labels.get("com.docker.compose.service") != SERVICE
                    or Path(labels.get("com.docker.compose.project.working_dir", "")).resolve() != root):
                raise RuntimeError("container identity mismatch")
            bindings = existing["HostConfig"].get("PortBindings", {}).get("8000/tcp", [])
            if not any(x["HostIp"] == "127.0.0.1" and int(x["HostPort"]) == port for x in bindings):
                raise RuntimeError("existing published port mismatch; explicit deployment update required")
        running = existing and existing["State"]["Running"]
        if not running:
            command(root, ["docker", "image", "inspect", service["image"]])
            command(root, ["docker", "compose", "up", "-d", "--no-build", "--no-recreate", SERVICE])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/control/status", timeout=2) as response:
                    value = json.loads(response.read())
                if not status_valid(value):
                    raise RuntimeError("invalid control status")
                if not running and (value["state"], value["residency"], value["active_requests"]) != ("unloaded", "not_resident", 0):
                    raise RuntimeError("new controller did not start unloaded")
                return value
            except Exception:
                time.sleep(.25)
        raise RuntimeError("control endpoint unavailable; existing environment was not restarted")


def main():
    root = Path(__file__).resolve().parents[2]
    try:
        prepare(root)
        return 0
    except Exception as exc:
        print(f"prepare failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
