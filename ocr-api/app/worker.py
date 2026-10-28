"""Exec boundary: only this child and its vLLM descendants initialize CUDA."""
import ctypes, os, signal, sys


def main():
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "cannot set parent-death signal")
    if os.getppid() != parent:
        raise SystemExit("controller exited before worker exec")
    os.execvp("vllm", ["vllm", "serve", *sys.argv[1:]])


if __name__ == "__main__":
    main()
