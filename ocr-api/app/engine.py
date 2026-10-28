import asyncio, hashlib, json, logging, os, signal, socket, sys, time
from pathlib import Path
import psutil

log = logging.getLogger("paddle.worker")


def start_ticks(pid):
    # Linux field 22 is invariant under WSL wall-clock/btime corrections.
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def verify_model(settings):
    root = Path(settings.model_path).resolve()
    manifest = json.loads(Path(settings.manifest_path).read_text())
    if manifest["revision"] != settings.revision or manifest["repo_id"] != "PaddlePaddle/PaddleOCR-VL-1.6":
        raise ValueError("selected model manifest identity mismatch")
    if len(manifest["files"]) != 17:
        raise ValueError("expected all 17 selected model files")
    for item in manifest["files"]:
        path = root / item["name"]
        if path.parent != root or not path.is_file() or path.is_symlink():
            raise ValueError("model manifest file missing or unsafe")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"model size mismatch: {item['name']}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"model hash mismatch: {item['name']}")
    config = json.loads((root / "config.json").read_text())
    if config.get("architectures") != ["PaddleOCRVLForConditionalGeneration"]:
        raise ValueError("unexpected model architecture")


class Worker:
    def __init__(self, settings, client):
        self.settings, self.client = settings, client
        self.process = None
        self.group = None
        self.tracked = {}
        self.drain_task = None
        self.start_time = 0
        self._stop_lock = asyncio.Lock()

    def snapshot_children(self):
        if self.process is None:
            return
        try:
            parent = psutil.Process(self.process.pid)
            if parent.pid in self.tracked and start_ticks(parent.pid) != self.tracked[parent.pid]:
                return
            processes = [parent, *parent.children(recursive=True)]
        except (psutil.NoSuchProcess, FileNotFoundError, ProcessLookupError):
            processes = []
        for proc in processes:
            try:
                self.tracked[proc.pid] = start_ticks(proc.pid)
            except (psutil.NoSuchProcess, FileNotFoundError, ProcessLookupError):
                pass

    def survivors(self):
        self.snapshot_children()
        found = []
        for pid, birth in list(self.tracked.items()):
            try:
                proc = psutil.Process(pid)
                if start_ticks(pid) == birth and proc.status() != psutil.STATUS_ZOMBIE:
                    found.append(proc)
            except (psutil.NoSuchProcess, FileNotFoundError, ProcessLookupError):
                pass
        # Catch descendants still in the owned process group after their parent exits.
        if self.group:
            for proc in psutil.process_iter(["pid", "status"]):
                try:
                    if (start_ticks(proc.pid) >= self.start_time - os.sysconf("SC_CLK_TCK")
                            and proc.info["status"] != psutil.STATUS_ZOMBIE
                            and os.getpgid(proc.pid) == self.group
                            and proc.pid not in self.tracked):
                        self.tracked[proc.pid] = start_ticks(proc.pid)
                        found.append(proc)
                except (psutil.NoSuchProcess, FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
        return found

    async def _drain(self):
        while True:
            data = await self.process.stdout.read(65536)
            if not data:
                return
            log.info("%s", data.decode(errors="replace").rstrip())

    async def start(self, generation):
        s = self.settings
        await asyncio.to_thread(verify_model, s)
        with socket.socket() as probe:
            probe.bind((s.worker_host, s.worker_port))
        self.tracked = {}
        self.start_time = int(time.clock_gettime(time.CLOCK_BOOTTIME) * os.sysconf("SC_CLK_TCK"))
        args = [sys.executable, "-m", "app.worker", s.model_path,
                "--host", s.worker_host, "--port", str(s.worker_port),
                "--served-model-name", s.model_id, "--dtype", "bfloat16",
                "--kv-cache-dtype", "auto", "--tensor-parallel-size", "1",
                "--max-model-len", str(s.max_model_len), "--max-num-seqs", str(s.max_active),
                "--max-num-batched-tokens", str(s.max_batched_tokens),
                "--gpu-memory-utilization", str(s.gpu_memory_utilization),
                "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0",
                "--trust-request-mm-kwargs", "--limit-mm-per-prompt", '{"image":1,"video":0}',
                "--mm-processor-kwargs", json.dumps({"min_pixels": s.min_pixels, "max_pixels": s.spotting_max_pixels})]
        if s.enforce_eager:
            args.append("--enforce-eager")
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", VLLM_NO_USAGE_STATS="1",
                   TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
        self.process = await asyncio.create_subprocess_exec(
            *args, env=env, start_new_session=True,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        self.group = self.process.pid
        self.snapshot_children()
        self.drain_task = asyncio.create_task(self._drain(), name=f"worker-log-{generation}")
        while True:
            self.snapshot_children()
            if self.process.returncode is not None:
                raise RuntimeError(f"vLLM exited with code {self.process.returncode}")
            try:
                models = await self.client.models()
                if not any(item["id"] == s.model_id for item in models["data"]):
                    raise RuntimeError("unexpected private served model")
                if self.process.returncode is not None:
                    raise RuntimeError("owned vLLM exited before readiness")
                return
            except (OSError, ConnectionError):
                pass
            except Exception as exc:
                # Network/startup failures are expected; a wrong model must fail.
                if isinstance(exc, RuntimeError):
                    raise
            await asyncio.sleep(.25)

    def signal_owned(self, proc, sig):
        fd = None
        try:
            fd = os.pidfd_open(proc.pid)
            if start_ticks(proc.pid) == self.tracked.get(proc.pid):
                signal.pidfd_send_signal(fd, sig)
        except (ProcessLookupError, FileNotFoundError):
            pass
        finally:
            if fd is not None:
                os.close(fd)

    async def stop(self):
        async with self._stop_lock:
            if self.process is None:
                return
            deadline = time.monotonic() + self.settings.unload_timeout
            processes = self.survivors()
            log.info("stopping owned worker pid=%s identities=%s", self.process.pid, self.tracked)
            for proc in processes:
                try:
                    self.signal_owned(proc, signal.SIGTERM)
                except (psutil.NoSuchProcess, FileNotFoundError, ProcessLookupError):
                    pass
            term_end = min(deadline - 1, time.monotonic() + self.settings.worker_term_grace)
            while self.survivors() and time.monotonic() < term_end:
                await asyncio.sleep(.1)
            for proc in self.survivors():
                try:
                    self.signal_owned(proc, signal.SIGKILL)
                except (psutil.NoSuchProcess, FileNotFoundError, ProcessLookupError):
                    pass
            while self.survivors() and time.monotonic() < deadline:
                await asyncio.sleep(.1)
            if self.survivors():
                raise TimeoutError("owned vLLM descendants did not terminate")
            await asyncio.wait_for(self.process.wait(), timeout=max(.1, deadline - time.monotonic()))
            if self.drain_task:
                await asyncio.wait_for(self.drain_task, timeout=max(.1, deadline - time.monotonic()))
            self.process, self.group = None, None
            self.tracked = {}

    def alive(self):
        return self.process is not None and self.process.returncode is None
