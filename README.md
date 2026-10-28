# paddle-ocr-docker-config

A Docker HTTP runtime that serves PaddleOCR-VL-1.6 BF16 using a **single vLLM engine**.
Each request runs one explicitly selected task on one image. The calling service handles tiling, cropping, content-type classification, additional recognition passes, and output correction.

## Setup and startup

Requirements: Linux/WSL2, Docker Compose with GPU support, an NVIDIA GPU, and a local copy of the model revision below.
Validated on an RTX 3090 with 24 GB VRAM. The runtime does not download models automatically.

- Model: PaddlePaddle/PaddleOCR-VL-1.6
- Revision: c5630abae1d940eafe0697512a0325494b02ab42
- Default model mount: MODEL_HOST_DIR=../models/ocr/tf → /models
- Model path: /models/PaddleOCR-VL-1.6
- All 17 file sizes and SHA256 hashes in configs/model_manifest.json are verified when loading the model.
- BF16 weights / KV cache dtype auto, no quantization / TP=1 / DP=1

For initial setup, create .env using .env.example as a reference. Preserve any existing .env file.

```bash
docker compose build
docker compose up
```

The default base image is pinned to the vLLM 0.31.0 registry digest in Dockerfile. Controller dependencies use the exact versions in requirements.txt without overriding the base image's vLLM/Torch versions. The base images, runtime images, settings, and results used for validation are recorded in [benchmarks/README.md](benchmarks/README.md).

The project follows the API directory and model mount conventions of qwen-asr/qwen-fa, along with the build and profile structure of vllm.

| Path | Purpose |
|---|---|
| ocr-api/app/ | Controller, model worker, and host-side prepare helper |
| Dockerfile | Root build context, as in vllm; container working directory /app |
| configs/common.env | Shared endpoint, input, and output limits |
| configs/models/paddleocr-vl-1.6.env | Model path, revision, processor, and context settings |
| .env | Host port, model parent directory, profile selection, and operational overrides |
| prepare-inferswap | Preserve running containers, start stopped containers, and exit with code 0 on success |

The controller starts on host port 8020 across all host interfaces (container port 8000), with the **model unloaded**. /health, /control/status, and ordinary /ocr requests do not load the model automatically. The internal vLLM port is not exposed to the host.

## Model control

During normal operation, InferSwap handles preparation, model loading/unloading, and GPU switching. The following direct calls are for standalone operation and validation.

```bash
curl --fail-with-body http://127.0.0.1:8020/control/status
curl --fail-with-body -X POST http://127.0.0.1:8020/control/load -H 'Content-Type: application/json' -d '{}'
curl --fail-with-body -X POST http://127.0.0.1:8020/control/unload -H 'Content-Type: application/json' -d '{}'
```

Successful status/control responses contain exactly four fields:

```json
{"state":"unloaded","residency":"not_resident","active_requests":0,"last_error":null}
```

Load/unload APIs are synchronous and wait for completion. Concurrent requests for the same operation share its result; requests for the opposite operation return 409 LIFECYCLE_CONFLICT. Unload returns 409 BUSY while inference or cancellation cleanup is active. Closing a control HTTP connection does not cancel an ongoing load/unload operation. If worker termination cannot be confirmed, the runtime remains failed/unknown and does not report the GPU as released.

`./prepare-inferswap` only prepares the control endpoint of an already built container. It preserves existing ready/active/failed states during normal operation and does not build images, download models, or load the model. If an existing endpoint fails validation, preparation fails without restarting or recreating the container.

## Single-task inference

POST /ocr, multipart/form-data:
file is one PNG/JPEG image, and payload is a JSON string as shown below. task is required; max_new_tokens is the only optional parameter.

| Task | Operation | Native output |
|---|---|---|
| spotting | Locate text and perform initial recognition | A string containing location tokens, with polygons and initial text that can be validated |
| ocr | Recognize text in the supplied image or crop | Raw model output |
| formula | Recognize mathematical expressions | Raw model output in LaTeX |
| table | Recognize table structure and content | Raw model output in OTSL |

```bash
curl --fail-with-body http://127.0.0.1:8020/ocr \
  -F 'file=@input.png;type=image/png' \
  -F 'payload={"task":"spotting","max_new_tokens":4096}'
```

Responses include request_id, model/revision, task, input hash/dimensions, native_text/text, technical_status, finish_reason/stop_reason, usage, and execution. Spotting responses also include parse_status, regions, and parse_errors. polygon_input_px uses coordinates in the **uploaded image**.

native_text preserves the string decoded by vLLM. The backend may omit EOS/end tokens from the returned output. technical_status=ok indicates technical completion of generation and structural processing; it does not establish recognition accuracy or quality acceptance. Output length limits, empty output, and spotting parse failures return HTTP 200 with incomplete status. Failed generations are not retried automatically. If native output has already been received when a timeout or stream failure occurs, it is preserved in the error's partial_result.

The runtime does not crop or tile images, classify content types automatically, run additional OCR after spotting, convert OTSL into cells/HTML, or correct LaTeX/text. The caller can compose one spotting request followed by N recognition requests as needed. If crops are already prepared, request the relevant task directly without spotting.

## Limits and cancellation

- Body limit: 32 MiB; one PNG/JPEG; at most 16,777,216 pixels, 8192 px per edge, and an aspect ratio of 200.
- Tasks: spotting/ocr/formula/table. Arbitrary prompts, URLs, mode, crop_policy, multiple images, and user-supplied processor kwargs are not accepted.
- max_new_tokens: a strict integer in 1..4096. Default: 4096.
- Default operating concurrency: 1. MAX_ACTIVE_REQUESTS accepts only 1/2/4, and the engine's max_num_seqs is set to the same value.
- HTTP errors: 429 for exhausted request slots, 503 for an unloaded model, 400 for invalid input, 413 for an oversized body, 415 for an unsupported format, 500 for inference failure, 504 for the internal request deadline, and 408 for the upload deadline.
- When the controller detects a client disconnect or reaches the request deadline, it cancels the corresponding private vLLM request. The request remains active until its termination is confirmed.
- If private HTTP cancellation provides no per-request termination acknowledgment, the runtime temporarily blocks new requests, waits for other normal requests to finish, and confirms that the entire engine is idle. Cancellation does not arbitrarily terminate other requests or unload the model.
- Recovery from a fatal or hung GPU engine is handled separately from normal request cancellation. If termination is uncertain, the runtime retains its failure state and responsibility for the running processes.

Revalidate the profile after changing MAX_ACTIVE_REQUESTS, pixel/token limits, context size, KV cache reservation, or eager execution settings. See [measured concurrency and limits](benchmarks/README.md).

## InferSwap integration

After registration, the request body is unchanged:

```bash
curl --fail-with-body 'http://127.0.0.1:8095/infer?model=paddle-ocr' \
  -F 'file=@prepared-crop.png;type=image/png' \
  -F 'payload={"task":"formula"}'
```

Use paddle-ocr as the logical ID, /ocr as inferencePath, http://127.0.0.1:8020 as baseURL on the same WSL host, and the absolute path to this repository's prepare-inferswap script in prepare.argv. Do not register a separate model for each task.

paddle-ocr was registered in InferSwap/config.vllm.yaml on the validation host, and integration validation completed. Preparation of a stopped container → model load → inference for all four tasks → external connection cancellation → OCR unload triggered by a Qwen request were all verified through port 8095. The existing settings for the three other models were preserved. Repository files do not automatically modify InferSwap settings on other hosts.

See the [connection and resource profile](tests/outputs/inferswap_connection.json), [registration fragment](benchmarks/inferswap_model.json), and [integration results](tests/outputs/20261028_inferswap_integration_03/integration_summary.json). InferSwap's backend status recheck was updated to handle delayed backend cleanup after an HTTP connection closes; the full Go test suite and router race checks passed.

## Validation

GPU-free tests cover contracts, races, input validation, cancellation, termination of real child process trees, and preservation of existing state during preparation:

```bash
docker run --rm --network none \
  -v "$PWD:/work:ro" -w /work -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 paddle-ocr-runtime:local \
  -m pytest -q -p no:cacheprovider
```

pytest is a development dependency. If it is absent from the selected base image, install it in a separate development environment.

Standalone validation runners are available in tests/check_api.py, tests/benchmark_concurrency.py, and tests/check_gpu_lifecycle.py. The GPU lifecycle runner checks the current local fixture manifest and existing runtime state, then attempts to restore the original residency after switching to Qwen. Each run requires a new output path; failed results are neither overwritten nor retried automatically. Fixtures can be replaced with a JSON manifest containing explicit image paths, tasks, and SHA256 hashes.

The InferSwap integration runner:

```bash
python3 tests/check_inferswap.py --out tests/outputs/<new-run>
```

This runner requires paddle-ocr/qwen-vlm registrations in InferSwap, an idle runtime state, and four fixed fixtures. It stops the OCR container, verifies that InferSwap starts it through preparation, and sends all inference and model-switching requests through port 8095. It does not call /control/load or /control/unload directly. Failed results are preserved in a new run directory, and the runner exits with a nonzero status.
