import asyncio, os, sys
from dataclasses import replace
import psutil
from app.config import Settings
from app.engine import Worker


def test_owned_process_tree_is_terminated_without_touching_unrelated_process(monkeypatch):
    async def scenario():
        unrelated = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time;time.sleep(60)")
        worker = Worker(replace(Settings(), worker_term_grace=.2, unload_timeout=3), None)
        worker.start_time = int(__import__("time").clock_gettime(__import__("time").CLOCK_BOOTTIME) * os.sysconf("SC_CLK_TCK"))
        worker.process = await asyncio.create_subprocess_exec(
            sys.executable, "-c",
            "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);print(p.pid,flush=True);time.sleep(60)",
            start_new_session=True, stdout=asyncio.subprocess.PIPE)
        worker.group = worker.process.pid
        child = int(await worker.process.stdout.readline())
        worker.snapshot_children()
        original_create_time = psutil.Process.create_time
        monkeypatch.setattr(psutil.Process, "create_time", lambda self, *a, **kw: original_create_time(self, *a, **kw) + 3600)
        await worker.stop()
        assert unrelated.returncode is None
        assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
        unrelated.terminate()
        await unrelated.wait()
    asyncio.run(scenario())
