# PaddleOCR runtime verification — 2026-10-28 KST

## Environment and build

- GPU-c47f1d0d-ea5d-4e6c-da7c-c9ad18b1b978, NVIDIA GeForce RTX 3090, 610.88, 24576 MiB
- PaddleOCR-VL-1.6 revision c5630abae1d940eafe0697512a0325494b02ab42; all17 selected model files verified on load.
- vLLM0.31.0 / Torch2.13.0+cu130 / Transformers5.17.0, BF16, KV auto, no quantization, TP=DP=1.
- Native FLASH_ATTN, eager; prefix caching and MM processor cache disabled.
- Standalone validation image: sha256:614feb4923d2ecc51da5543762a79b525cf6af87cdcca06cffbb8fa22e4d55b2.
- Standalone build used the verified local vllm-docker-runtime image sha256:936f9a6d71feb5179f0bed3275c28a0eb142d743a84e72e7e39d6ecd8c9b8402. Dockerfile's default registry base is pinned separately.
- [Release source/image manifest](../tests/outputs/20261028_release_manifest.json), [measurements](measurements.json).

## API and CPU verification

The standalone release passed36 CPU tests: strict payload/control bodies, chunked body limits, shared load/unload lifetime, both admission/unload race orders, active/BUSY protection, cancel-before-start, multi-request cancellation, uncertain cleanup convergence, stream partial output retention, prepare preservation, and actual owned child process cleanup without killing an unrelated process.

[JUnit](../tests/outputs/20261028_cpu_release.xml) records that standalone CPU result; the structure/prepare update passes38 tests below. The process test shifts psutil wall-clock creation times to cover the shutdown regression. Production ownership uses invariant /proc start ticks and pidfd signals.

## Measured concurrency

Same four frozen development images/tasks, four warmup calls per profile, then two identical four-request windows at C1/2/4. max_num_seqs matches C. [Fixtures](../tests/runtime_fixtures.json), [raw ledger](../tests/outputs/20261028_runtime_validation_01/ledger.json), [windows](../tests/outputs/20261028_runtime_validation_01/concurrency.json).

| C | Four-request seconds, two windows | Mean requests/s | Mean completion tokens/s | Host GPU peak MiB |
|---:|---|---:|---:|---:|
| 1 | 5.966 / 6.004 | 0.668 | 65.3 | 10038 |
| 2 | 4.659 / 4.513 | 0.872 | 85.3 | 10034 |
| 4 | 3.285 / 3.325 | 1.210 | 115.9 | 9979 |

Actual controller max active1/2/4 was observed. KV is preallocated, so host memory does not scale proportionally with C. These are mixed-task small-workload measurements, not universal OCR throughput or a stable production p95 estimate. Each level has only8 latency samples; descriptive p50/p95 are in measurements.json.

**Operating default: C1.** C2 changed Spotting native output, and C4 changed Spotting/OCR native output compared with the repeatable C1 reference. Formula/table output matched. The selection rule chooses the fastest level that repeatedly reproduces every reference native output; this selects1. C2/4 are technically runnable configuration options, with output differences preserved rather than judged as quality improvements. Changing C requires a matching resource profile and output review.

## Lifecycle, cancellation and bounds

[Targeted GPU run](../tests/outputs/20261028_runtime_validation_03/summary.json) passed:

- Corrected C4 cancellation/shutdown; cancelling one request did not terminate the unrelated request or unload the model. Active stayed protected until engine cleanup; active unload returned409 BUSY.
- Three C1 load/infer/unload rounds. Load45.05–46.52s, unload0.51–0.61s; no owned vLLM processes remained after any round. [Cycles](../tests/outputs/20261028_runtime_validation_03/cycles.json).
- 8192×2048 synthetic dense table input, exactly4096 completion tokens, HTTP200 incomplete/length, about62.45s. This validates the bound/termination behavior, not table transcription accuracy. [Result](../tests/outputs/20261028_runtime_validation_03/maximum_image.json).
- A one-token output limit returned incomplete/length.
- Real request deadline cancellation converged while preserving ready/resident model state.
- Restart of a model-resident container cleaned workers and booted unloaded. [Restart](../tests/outputs/20261028_runtime_validation_03/restart.json).
- Maximum observed controller status latency in this run: 1.077s, including load/generation/abort/unload.

[Final partial-output GPU run](../tests/outputs/20261028_runtime_validation_04/summary.json) passed. A15s deadline preserved1754 native characters and input hash/task/request ID in the504 error, then converged active and unloaded successfully. [Raw response](../tests/outputs/20261028_runtime_validation_04/partial_timeout.json).

Final release changes after that GPU run only adjust fatal-error partial forwarding/buffer cleanup and oversized control-body400 handling. These branches are CPU-tested; the final image's unloaded controller/prepare/HTTP validation was checked. Normal inference/engine options and timeout path are unchanged. Exact source diffs and GPU image IDs are retained in the release manifest/measurements.

## Resource profile and registration fragment

[inferswap_model.json](inferswap_model.json) is a measured registration fragment for the existing InferSwap schema. It is now registered on this host, as verified in the integration section below.

- Host observed targeted peak: 10229MiB, conservative pre-load baseline1451MiB.
- Both load/inference caps use the higher observed delta plus1GiB sampling margin: 10278141952bytes. This deliberately includes the higher C4 startup observation while the registered operating concurrency is1.
- Largest positive selected-round unload delta: 1MiB, with no remaining model processes. Host GUI/background noise is not model residency. Residual cap includes64MiB margin: 68157440bytes; it is not fabricated as zero.
- Limits bind revision/image/config, maxConcurrency1, one image, max16,777,216 input pixels/8192edge, processor caps, max4096 output, context8192, and request deadline120s.
- Recommended InferSwap load wait180s covers runtime load deadline120s plus failure cleanup30s and transport margin. Unload45s covers the internal30s deadline. These timeout recommendations are not measured load durations.

The sampling margins and tested fixtures do not establish content quality, every extreme aspect-ratio combination, or a general performance guarantee. Maximum-grid/multimodal compatibility is additionally constrained by the pinned processor and runtime dimensions. Re-measure after changing pixel/token/context/KV/eager/concurrency settings.

## Preserved failures and budget

- Run01: all36 normal calls plus a two-request cancellation case were recorded; C4 unload failed because process identity/termination confirmation did not converge. The failure was not relabelled as pass. [Summary](../tests/outputs/20261028_runtime_validation_01/summary.json), [shutdown log](../tests/outputs/20261028_runtime_validation_01/failed_shutdown_server.log). Initial source snapshots remain intact.
- Fix: stable Linux start ticks + pidfd, instead of wall-clock creation timestamps; failed/unknown cleanup keeps supervising until actual termination is confirmed.
- Run02: one sequence reservation, stopped by the observer's3s cold private-metrics timeout. The runtime unloaded and Qwen was restored. [Summary](../tests/outputs/20261028_runtime_validation_02/summary.json). The corrected verifier explicitly reserves a warmup before observing cancellation.
- Run03:12 reservations; targeted lifecycle/boundary validation passed.
- Run04:2 reservations; final partial-output GPU path passed.
- Total runtime-stage reservations: 53 of56. No automatic inference retries. Earlier backend-only probes have their separate ledgers.

At the end of the standalone stage, Qwen was restored ready/resident/active0 and the OCR controller remained unloaded; ASR/FA and InferSwap configuration had been preserved. The following separately authorized stage performs actual InferSwap integration. ExtracTube/T5 quality adoption remains separate.

## Structure alignment and actual InferSwap integration

The current image is sha256:15eea8969f605ec39e40c3c57c1886a73b647186fade9c37211f1adad7ab3208, built successfully from the Dockerfile's pinned official vLLM0.31.0 digest. [Current release manifest](../tests/outputs/20261028_inferswap_integration_03/release_manifest.json) binds runtime/verifier/InferSwap source, image and config evidence. Earlier manifests remain historical.

| Common pattern | PaddleOCR choice | Reference |
|---|---|---|
| API code below a service directory, /app working path | ocr-api/app → /app/app | qwen-asr and qwen-fa |
| Upper model directory at /models | MODEL_HOST_DIR=../models/ocr/tf, read-only | qwen-asr and qwen-fa |
| Root Dockerfile, common/model env files, cache volume, /health | configs/common.env + configs/models/paddleocr-vl-1.6.env | vllm |
| Quiet prepare exit0; preserve running container; start stopped controller without build/recreate/load | root shell wrapper + standard-library helper, Compose working-dir identity | all three; vllm helper ownership pattern |

[Settings comparison](../tests/outputs/20261028_inferswap_integration_01/settings_equivalence.json) shows every inference/engine/limit setting unchanged; only the manifest path moves from /runtime/configs to /app/configs. OCR-specific payloads, loopback publication, manifest integrity and worker/cancel ownership remain intentional.

**Actual gateway verification passed**, using only POST8095 /infer?model=… for inference and model-transition triggers. No direct /control/load or /control/unload POSTs were made by the integration verifier.

- A stopped OCR container was started by InferSwap prepare, then loaded for the first request. The container ID stayed the same.
- spotting/ocr/formula/table each returned200, finish_reason=stop, one sequence, the exact uploaded image SHA256. These four development crops are functional evidence, not content-quality acceptance.
- Closing the proxy connection cancelled the active table request; active1→0 in0.313s while the model stayed ready/resident.
- The next qwen-vlm request caused InferSwap to unload OCR, observe GPU release and load Qwen. OCR's controller stayed running, unloaded/not_resident/active0, with no owned vLLM workers. Qwen returned200 and was restored ready/resident/active0.
- The operating profile stays C1, BF16, one model instance, with the measured bounds above.

[Summary](../tests/outputs/20261028_inferswap_integration_03/integration_summary.json), [proxy cancellation](../tests/outputs/20261028_inferswap_integration_03/proxy_cancel.json), [InferSwap transition events](../tests/outputs/20261028_inferswap_integration_03/inferswap_ocr_events.json), [Qwen response](../tests/outputs/20261028_inferswap_integration_03/qwen_switch_response.json), [sampled states/VRAM](../tests/outputs/20261028_inferswap_integration_03/integration_samples.json), [connection artifact](../tests/outputs/inferswap_connection.json).

### Integration fixes and preserved failures

- Integration01 failed before inference: a stopped-container prepare did a host bind probe that rejected a harmless TIME_WAIT socket. The helper now lets Docker Compose check actual port ownership, like the sibling projects. Compose identity and running-container preservation remain enforced. [Failed summary](../tests/outputs/20261028_inferswap_integration_01/integration_summary.json).
- Integration02 passed prepare/load, four tasks and proxy cancellation, but Qwen switching returned504 QUEUE_TIMEOUT. InferSwap read backend active1 once after HTTP cancellation and never noticed later active0. The old server's shutdown also reached its deadline because that slot remained protected. All raw responses and logs remain unchanged. [Failed summary](../tests/outputs/20261028_inferswap_integration_02/integration_summary.json).
- InferSwap now rechecks backend status every100ms while the same quiet request generation is active or its status query fails. A new dispatch invalidates the old observation chain; idle or shutdown stops polling. Active/BUSY protections and generation-based idle proof remain intact.
- Two regression cases reproduced the issue before the fix: asynchronous backend cleanup and an initial status error. Both pass afterward, including no continued polling once idle. [Before result](../tests/outputs/20261028_inferswap_integration_03/regression_before.txt), [whole Go suite](../tests/outputs/20261028_inferswap_integration_03/go_all.txt), [router race detector](../tests/outputs/20261028_inferswap_integration_03/go_router_race.txt).
- Current OCR CPU suite:38 passed, including foreign-worktree identity and stopped-container prepare. [JUnit](../tests/outputs/20261028_inferswap_integration_01/cpu_prepare_fix.xml).
- Integration03 reserved5 OCR requests (four completed tasks + one disconnect), within8, and used2 Qwen gateway requests including recovery, within2. The earlier failed integration attempts used1 unexecuted OCR reservation in01,5 OCR reservations and1 timed-out Qwen request in02. These are separate integration budgets from the prior53/56 standalone runtime reservations. No automatic inference retries.

The host's InferSwap/config.vllm.yaml has one added paddle-ocr registration; the previous three registrations are unchanged. InferSwap is left running in a visible WSL terminal. Its user-provided run.sh is preserved. OCR is healthy/unloaded, Qwen ready/resident, ASR/FA unloaded. Local implementation documents remain exactly two ignored MD files; no local README was added.
