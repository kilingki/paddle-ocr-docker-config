"""Read-only clients and immutable request artifacts for independent GPU checks."""
from pathlib import Path
import hashlib, json, mimetypes, threading, time, urllib.request, urllib.error, uuid

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def json_request(base, path, data=None, timeout=10):
    request = urllib.request.Request(base + path,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def load_fixtures(path):
    manifest = json.loads(Path(path).read_text())
    fixtures = []
    for item in manifest:
        file = ROOT.parent / item["path"]
        if hashlib.file_digest(file.open("rb"), "sha256").hexdigest() != item["sha256"]:
            raise ValueError("fixture hash mismatch: " + str(file))
        fixtures.append({**item,"file":file})
    return fixtures


def multipart(fixture, max_tokens=4096):
    boundary = "paddle-" + uuid.uuid4().hex
    payload = json.dumps({"task":fixture["task"],"max_new_tokens":max_tokens})
    content = fixture["file"].read_bytes()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"payload\"\r\n\r\n{payload}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.png\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, {"Content-Type":"multipart/form-data; boundary=" + boundary}


class Ledger:
    def __init__(self, out, budget=48):
        self.out, self.budget = Path(out), budget
        if (self.out / "ledger.json").exists():
            raise RuntimeError("existing ledger preserved; use a fresh output directory")
        self.lock = threading.Lock()
        self.calls = []

    def reserve(self, fixture, label):
        with self.lock:
            if len(self.calls) >= self.budget:
                raise RuntimeError("bounded sequence reservation budget exhausted")
            row = {"index":len(self.calls), "task":fixture["task"], "fixture_sha256":fixture["sha256"],
                   "label":label, "reserved_at":time.monotonic()}
            self.calls.append(row)
            write(self.out/"ledger.json",self.calls)
            return row

    def infer(self, base, fixture, label, max_tokens=4096):
        row = self.reserve(fixture,label)
        start = time.monotonic()
        body, headers = multipart(fixture,max_tokens)
        req = urllib.request.Request(base+"/ocr",data=body,headers=headers)
        try:
            with urllib.request.urlopen(req,timeout=180) as response:
                result = {"http_status":response.status,"body":json.loads(response.read())}
        except urllib.error.HTTPError as exc:
            result = {"http_status":exc.code,"body":json.loads(exc.read())}
        except Exception as exc:
            result = {"http_status":None,"error":repr(exc)}
        result["seconds"] = time.monotonic()-start
        with self.lock:
            row.update(result)
            write(self.out/f"{row['index']:03d}_response.json",row)
            write(self.out/"ledger.json",self.calls)
        return row
