# paddle-ocr-docker-config

PaddleOCR-VL-1.6 BF16를 **단일 vLLM engine**으로 실행하는 Docker HTTP runtime입니다.
한 요청은 이미지 한 장에 명시한 task 하나를 실행합니다. tile/crop·자료 유형 판단·추가 판독·결과 보정은 호출 서비스가 담당합니다.

## 준비와 실행

필요 환경: Linux/WSL2, Docker Compose GPU 지원, NVIDIA GPU, 아래 revision의 로컬 모델.
RTX 3090 24GB에서 검증했습니다. 모델은 자동 다운로드하지 않습니다.

- 모델: PaddlePaddle/PaddleOCR-VL-1.6
- revision: c5630abae1d940eafe0697512a0325494b02ab42
- 기본 모델 마운트: MODEL_HOST_DIR=../models/ocr/tf → /models
- 모델 경로: /models/PaddleOCR-VL-1.6
- configs/model_manifest.json의 17개 파일 크기/SHA256을 load 때 검증
- 가중치 BF16 / KV auto, 양자화 없음 / TP=1 / DP=1

최초 설정 때 .env.example을 참고해 .env를 작성합니다. 기존 .env는 보존합니다.

```bash
docker compose build
docker compose up
```

기본 base는 Dockerfile의 vLLM0.31.0 registry digest에 고정되어 있습니다. controller 의존성은 requirements.txt의 정확한 버전을 사용하며 vLLM/Torch 버전을 덮어쓰지 않습니다. 실제 검증 base/image·설정·결과는 [benchmarks/README.md](benchmarks/README.md)에 기록합니다.

구조는 qwen-asr/qwen-fa의 API 디렉터리·모델 마운트와 vllm의 빌드·profile 구성을 따릅니다.

| 위치 | 역할 |
|---|---|
| ocr-api/app/ | controller·모델 worker·호스트 prepare helper |
| Dockerfile | vllm과 같은 루트 build context, 컨테이너 작업 경로 /app |
| configs/common.env | 공통 endpoint·입력·출력 한도 |
| configs/models/paddleocr-vl-1.6.env | 모델 경로·revision·processor/context 설정 |
| .env | 호스트 포트·모델 상위 디렉터리·profile 선택·운영 override |
| prepare-inferswap | 실행 중 컨테이너 보존, 정지 컨테이너 기동, 성공 시 exit0 |

controller는 모든 호스트 인터페이스의 8020 포트(컨테이너 내부 8000)에서 시작하고 **모델은 unloaded**입니다. /health·/control/status·일반 /ocr 요청은 모델을 자동 load하지 않습니다. 내부 vLLM 포트는 호스트에 공개하지 않습니다.

## 모델 제어

일반 운영에서는 InferSwap이 prepare/load/unload와 GPU 전환을 담당합니다. 다음 직접 호출은 독립 운영·검증용입니다.

```bash
curl --fail-with-body http://127.0.0.1:8020/control/status
curl --fail-with-body -X POST http://127.0.0.1:8020/control/load -H 'Content-Type: application/json' -d '{}'
curl --fail-with-body -X POST http://127.0.0.1:8020/control/unload -H 'Content-Type: application/json' -d '{}'
```

정상 status/control 성공 응답은 정확히 네 필드입니다.

```json
{"state":"unloaded","residency":"not_resident","active_requests":0,"last_error":null}
```

load/unload는 완료까지 기다리는 동기 API입니다. 중복 같은 작업은 같은 결과를 공유하고 반대 작업은409 LIFECYCLE_CONFLICT입니다. 활성 추론·취소 정리가 있으면 unload는409 BUSY입니다. control HTTP 연결 종료는 진행 중 load/unload를 취소하지 않습니다. 실제 worker 종료를 확인하지 못하면 failed/unknown으로 남겨 GPU가 해제됐다고 보고하지 않습니다.

`./prepare-inferswap`은 build된 컨테이너의 control endpoint만 준비합니다. 정상 기존 ready/active/failed 상태를 보존하고 build/download/model load를 하지 않습니다. 기존 endpoint 확인 실패 시 restart/recreate하지 않고 실패합니다.

## 단일 task 추론

POST /ocr, multipart/form-data:
file은 PNG/JPEG 한 장, payload는 아래 JSON 문자열입니다. task는 필수이며 max_new_tokens만 선택할 수 있습니다.

| task | 실행 | native 출력 |
|---|---|---|
| spotting | 문자 위치와 최초 판독 | 위치 token을 포함한 문자열, 검증 가능한 polygon/최초 text |
| ocr | 전달된 이미지·crop 문자 판독 | 모델 원문 |
| formula | 수식 판독 | 모델 원문 LaTeX |
| table | 표 구조·내용 판독 | 모델 원문 OTSL |

```bash
curl --fail-with-body http://127.0.0.1:8020/ocr \
  -F 'file=@input.png;type=image/png' \
  -F 'payload={"task":"spotting","max_new_tokens":4096}'
```

응답에는 request_id, model/revision, task, 입력 hash/크기, native_text/text, technical_status, finish_reason/stop_reason, usage, execution이 있습니다. spotting에는 parse_status, regions, parse_errors가 추가됩니다. polygon_input_px는 **업로드 이미지** 좌표입니다.

native_text는 vLLM decode 문자열을 보존합니다. EOS/종료 token은 backend가 반환에서 제외할 수 있습니다. technical_status=ok는 생성/구조 처리의 기술적 완료이며 문자 정확도나 품질 채택이 아닙니다. 길이 상한·비어 있는 출력·Spotting parse 실패는200 incomplete로 반환합니다. 실패한 생성은 자동 재시도하지 않습니다. timeout/stream 실패에 이미 받은 native 원문이 있으면 오류의 partial_result에 보존합니다.

runtime은 crop·tile·자동 유형 분류·Spotting 뒤 추가 OCR·OTSL 셀/HTML 변환·LaTeX/문자 교정을 수행하지 않습니다. 외부에서 Spotting 1회 뒤 필요한 판독 N회를 조합합니다. 이미 crop이 준비되어 있으면 spotting 없이 해당 task를 직접 요청합니다.

## 제한과 취소

- body32MiB, 단일 PNG/JPEG, 최대16,777,216 pixels·한 변8192px·종횡비200.
- task: spotting/ocr/formula/table. 자유 prompt·URL·mode·crop_policy·복수 이미지·사용자 processor kwargs는 받지 않습니다.
- max_new_tokens: 엄격한 정수1..4096. 기본4096.
- 초기 운영 동시성1. MAX_ACTIVE_REQUESTS는1/2/4만 허용하며 engine max_num_seqs도 같은 값으로 설정합니다.
- 슬롯 초과429, 미로드503, 입력400, body413, 형식415, 추론 실패500, 내부 deadline504, 업로드 deadline408.
- controller가 클라이언트 disconnect를 감지하거나 요청 deadline에 도달하면 해당 private vLLM 요청을 취소합니다. 실제 종료 확인 전 active를 유지합니다.
- private HTTP 취소의 개별 종료 ACK가 없으면 새 접수를 잠시 막고 나머지 정상 요청이 끝난 뒤 engine 전체 idle을 확인합니다. 취소가 다른 요청을 임의로 중단하거나 모델을 unload하지 않습니다.
- GPU engine fatal/hang의 복구는 정상 요청 취소와 구분합니다. 종료 불명이면 fail 상태와 실행 책임을 유지합니다.

MAX_ACTIVE_REQUESTS·pixel/token·context·KV 예약·eager 설정을 바꾸면 profile을 다시 검증합니다. [동시성 실측과 제한](benchmarks/README.md)을 확인하세요.

## InferSwap 연결

등록 후 호출 body는 동일합니다.

```bash
curl --fail-with-body 'http://127.0.0.1:8095/infer?model=paddle-ocr' \
  -F 'file=@prepared-crop.png;type=image/png' \
  -F 'payload={"task":"formula"}'
```

logical ID는 paddle-ocr, inferencePath는 /ocr, baseURL은 같은 WSL host 기준 http://127.0.0.1:8020, prepare.argv는 이 저장소 prepare-inferswap의 절대 경로입니다. task마다 모델을 등록하지 않습니다.

이 호스트의 InferSwap/config.vllm.yaml에 paddle-ocr를 등록하고 실제 통합 검증을 마쳤습니다. 정지 컨테이너 prepare → load → 네 task 추론 → 외부 연결 취소 → Qwen 요청에 따른 OCR unload를 모두 8095 경유로 확인했습니다. 실행 중이던 기존 세 모델의 설정은 보존했습니다. 저장소 파일이 다른 호스트의 InferSwap 설정을 자동 변경하지는 않습니다.

[접속·자원 profile](tests/outputs/inferswap_connection.json), [등록 fragment](benchmarks/inferswap_model.json), [통합 결과](tests/outputs/20261028_inferswap_integration_03/integration_summary.json)를 참고하세요. HTTP 종료 뒤 backend 정리가 늦어지는 경우를 처리하도록 InferSwap의 backend 상태 재확인을 수정했고, 전체 Go 테스트와 router race 검증을 통과했습니다.

## 검증

GPU 없는 계약·경합·입력·취소·실제 자식 process tree 종료·prepare 보존 테스트:

```bash
docker run --rm --network none \
  -v "$PWD:/work:ro" -w /work -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 paddle-ocr-runtime:local \
  -m pytest -q -p no:cacheprovider
```

pytest는 개발용입니다. 사용 base에 없으면 별도 개발 환경에 설치합니다.

독립 검증 실행기는 tests/check_api.py, benchmark_concurrency.py, check_gpu_lifecycle.py에 있습니다. GPU lifecycle 실행기는 현재 local fixture manifest와 기존 runtime 상태를 검사하고, Qwen 전환 후 원래 residency 복원을 시도합니다. 새 run 출력 경로가 필요하며 실패를 덮어쓰거나 자동 재시도하지 않습니다. fixture는 명시적인 이미지 경로·task·SHA256 JSON으로 교체할 수 있습니다.


실제 InferSwap 통합 실행기:

```bash
python3 tests/check_inferswap.py --out tests/outputs/<new-run>
```

InferSwap의 paddle-ocr/qwen-vlm 등록과 유휴 상태, 네 고정 fixture가 필요합니다. 이 실행기는 OCR 컨테이너를 정지한 뒤 InferSwap의 prepare 기동을 확인하고, 모든 추론과 모델 전환을 8095로 요청합니다. 직접 /control/load·unload를 호출하지 않습니다. 실패 결과는 새 run에 보존하고 비0으로 종료합니다.
