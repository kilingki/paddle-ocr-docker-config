ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
FROM ${VLLM_BASE_IMAGE}
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_NO_USAGE_STATS=1 HF_HOME=/cache/huggingface VLLM_CACHE_ROOT=/cache/vllm \
    TORCH_HOME=/cache/torch XDG_CACHE_HOME=/cache/xdg
WORKDIR /app
COPY ocr-api/requirements.txt /app/requirements.txt
# Retain the base image's vLLM/Torch pins; install controller packages only.
RUN python3 -m pip install --no-cache-dir --no-deps -r /app/requirements.txt
COPY ocr-api/app/ /app/app/
COPY configs/ /app/configs/
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
CMD []
