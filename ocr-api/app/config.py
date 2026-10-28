from dataclasses import dataclass
from pathlib import Path
import os

REVISION = "c5630abae1d940eafe0697512a0325494b02ab42"
PROMPTS = {"spotting": "Spotting:", "ocr": "OCR:",
           "formula": "Formula Recognition:", "table": "Table Recognition:"}


@dataclass(frozen=True)
class Settings:
    model_path: str = "/models/PaddleOCR-VL-1.6"
    manifest_path: str = "/app/configs/model_manifest.json"
    model_id: str = "paddle-ocr"
    revision: str = REVISION
    worker_host: str = "127.0.0.1"
    worker_port: int = 18000
    max_active: int = 1
    max_inflight_uploads: int = 4
    max_body_bytes: int = 33554432
    max_image_pixels: int = 16777216
    max_image_edge: int = 8192
    max_new_tokens: int = 4096
    min_pixels: int = 112896
    spotting_max_pixels: int = 1605632
    reading_max_pixels: int = 1003520
    max_model_len: int = 8192
    max_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.35
    load_timeout: float = 120
    unload_timeout: float = 30
    request_timeout: float = 120
    cancel_timeout: float = 30
    upload_timeout: float = 30
    worker_term_grace: float = 10
    health_interval: float = 5
    enforce_eager: bool = True

    def __post_init__(self):
        integers = ("worker_port", "max_active", "max_inflight_uploads", "max_body_bytes",
                    "max_image_pixels", "max_image_edge", "max_new_tokens", "min_pixels",
                    "spotting_max_pixels", "reading_max_pixels", "max_model_len", "max_batched_tokens")
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("load_timeout", "unload_timeout", "request_timeout", "cancel_timeout",
                     "upload_timeout", "worker_term_grace", "health_interval"):
            if not 0 < getattr(self, name) < float("inf"):
                raise ValueError(f"{name} must be positive and finite")
        if self.max_active not in (1, 2, 4):
            raise ValueError("MAX_ACTIVE_REQUESTS must be 1, 2, or 4")
        if self.worker_host != "127.0.0.1" or self.worker_port > 65535:
            raise ValueError("worker must bind an internal loopback port")
        if self.worker_term_grace >= self.unload_timeout:
            raise ValueError("TERM grace must leave time for KILL/reap")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("GPU_MEMORY_UTILIZATION must be between 0 and 1")
        if self.max_new_tokens > 4096 or self.max_new_tokens >= self.max_model_len:
            raise ValueError("invalid output/context limit")
        if self.min_pixels > min(self.spotting_max_pixels, self.reading_max_pixels):
            raise ValueError("processor pixel limits are inconsistent")
        if self.revision != REVISION or self.model_id != "paddle-ocr":
            raise ValueError("this deployment fixes the selected model identity")

    @property
    def worker_url(self):
        return f"http://{self.worker_host}:{self.worker_port}"

    @classmethod
    def from_env(cls):
        defaults = cls()
        names = {"max_active": "MAX_ACTIVE_REQUESTS", "worker_port": "WORKER_PORT"}
        values = {}
        for name in cls.__dataclass_fields__:
            env_name = names.get(name, name.upper())
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            default = getattr(defaults, name)
            if type(default) is bool:
                if raw.lower() not in ("true", "false", "1", "0"):
                    raise ValueError(f"{env_name} must be true or false")
                values[name] = raw.lower() in ("true", "1")
            elif cls.__dataclass_fields__[name].type is int:
                values[name] = int(raw)
            elif cls.__dataclass_fields__[name].type is float:
                values[name] = float(raw)
            else:
                values[name] = raw
        return cls(**values)
