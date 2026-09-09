# FunASR 배포 선택 매트릭스

## Transformers로 Nano 시작하기

중국어·영어·일본어 전사는 [Transformers 5.17.0 가이드(영문)](./transformers_native.md)와 공식 `FunAudioLLM/Fun-ASR-Nano-2512-hf` checkpoint로 시작할 수 있습니다. CPU 예제가 있으며 toolkit과 서비스 경로는 별도입니다. 네이티브 출력은 텍스트이며 타임스탬프·화자·HTTP 서버를 추가하지 않습니다.

제품, 데모, 벤치마크, 내부 워크플로에 맞는 가장 짧은 배포 경로를 고르기 위한 가이드입니다. 먼저 요구를 만족하는 최소 구성에서 시작하고, throughput, latency, integration 요구가 명확해질 때 더 무거운 runtime으로 이동하세요.

## 빠른 결정 표

| Path | 적합한 용도 | 시작 문서 | 운영 메모 |
|---|---|---|---|
| Colab notebook | 브라우저 smoke test, 첫 평가, 공유 가능한 demo | [Colab 빠른 시작](../examples/colab/README_ko.md) | 로컬 환경이 필요 없습니다. 첫 실행은 모델을 다운로드하며 GPU runtime이 더 빠릅니다. |
| Python API | Notebook, offline job, 첫 model evaluation | [README quick start](../README_ko.md#빠른-시작) | 가장 단순한 경로입니다. batching, retry, file 관리는 호출 측에서 담당합니다. |
| OpenAI 호환 API | Private speech API, Agent, Dify/LangChain/AutoGen style clients | [OpenAI API example](../examples/openai_api/README_ko.md) | OpenAI audio API를 이미 지원하는 앱에 가장 쉽게 연결됩니다. |
| Docker Compose API | 재현 가능한 local smoke test, 작은 internal service | [OpenAI API Docker docs](../examples/openai_api/README_ko.md#docker-배포) | 기본은 CPU입니다. CUDA를 쓰기 전에 CUDA-capable image로 조정하세요. |
| Kubernetes API | Cluster service용 internal speech API | [Kubernetes template](../examples/openai_api/kubernetes/) | private `ClusterIP`부터 시작합니다. 범위를 넓히기 전에 auth, TLS, network policy, GPU scheduling을 추가하세요. |
| Runtime WebSocket service | Live captions, meeting, call-center stream | [Runtime service docs](../runtime/readme.md) | partial result, endpointing, long-lived audio stream이 중요할 때 사용합니다. |
| vLLM acceleration | Fun-ASR-Nano의 native 파일 전사 또는 split-engine 디코딩 | [공식 native 검증(영문)](./vllm_official_native_validation.md), [split-engine guide](./vllm_guide.md) | 두 경로는 checkpoint와 API가 다릅니다. 비자기회귀 Paraformer에는 적용되지 않습니다. |
| MOSS-Transcribe-Diarize | 긴 다중 화자 transcription, timestamp, speaker label | [Third-party MOSS guide](./moss_transcribe_diarize.md) | OpenMOSS의 Apache-2.0 model이며 FunASR `AutoModel`에 통합되어 있습니다. local HF(`backend="hf"`), vLLM(`backend="vllm"`) 또는 SGLang Omni(`backend="sglang"`)를 선택할 수 있습니다. model의 공개 및 유지 관리는 계속 OpenMOSS가 담당합니다. |
| MCP server | Claude/Cursor/desktop agent speech tool | [MCP example](../examples/mcp_server/) | ASR 결과를 local tool로 Agent에 전달할 때 유용합니다. |
| Subtitle generator | 긴 audio/video에서 SRT/VTT 생성 | [Subtitle example](../examples/subtitle/) | readability가 중요하면 verbose segment와 speaker label을 사용합니다. |
| Batch ASR script | Archive, meeting, dataset, 반복 offline run | [Batch example](../examples/batch_asr_improved.py) | production에서는 queue, manifest, retry log를 추가하세요. |

## 자주 쓰는 선택

### Fun-ASR-Nano를 vLLM으로 실행하고 싶다

- **Split-engine**: 기본 모델 `FunAudioLLM/Fun-ASR-Nano-2512`를 사용하며, FunASR가 오디오를 처리하고 LLM decoder를 vLLM에서 실행합니다. [split-engine guide](./vllm_guide.md)를 참고하세요.
- **Native**: vLLM 자체가 오디오 모델을 실행합니다. 공식 checkpoint는 `FunAudioLLM/Fun-ASR-Nano-2512-vllm`입니다. 기본 모델용 설정이나 checkpoint와 서로 바꾸어 사용하지 마세요.

[공식 native 검증 기록(영문)](./vllm_official_native_validation.md)은 모델 revision
`a4362c943d48951f98ca2a62181cc028970270c5`, vLLM 0.27.1, 기존 H100 환경에서의 기능 확인입니다.
모델 revision은 FunASR 패키지 버전이 아닙니다. 의존성을 포함한 준비 및 실행 절차는 검증 기록을 참고하세요.

확인한 API는 파일 입력의 `/v1/audio/transcriptions`입니다. `/v1/realtime`, 클린 설치, 긴 오디오, 화자 분리, 프로덕션 처리 용량은 검증 대상이 아닙니다. 이를 FunASR SDK나 다른 checkpoint의 검증으로 간주하지 말고, 실제 운영 오디오와 부하로 별도 평가하세요.

### 5분 안에 FunASR을 시험하고 싶다

브라우저만으로 확인하려면 [Colab 빠른 시작](../examples/colab/README_ko.md)을 사용하세요. 로컬에서 작업하려면 README의 Python API부터 시작합니다. 어떤 모델을 고를지 고민된다면 [모델 선택 가이드](./model_selection_ko.md)를 참고하세요.

### Cloud transcription의 local replacement가 필요하다

OpenAI 호환 API를 사용하세요. 주요 진입점은 다음과 같습니다.

- `/v1/audio/transcriptions`: 파일 전사
- `/v1/models`: 모델 목록
- `/health`: 상태 확인
- Swagger docs: API 확인

먼저 `sensevoice`로 smoke test를 실행하고 기존 SDK나 HTTP client를 [OpenAI API example](../examples/openai_api/README_ko.md)에 맞춰 연결하세요.

### 재현 가능한 container demo가 필요하다

Docker Engine과 Docker Compose plugin을 준비한 뒤 **FunASR 저장소 루트**에서 로컬 SenseVoice CPU service를 시작하세요. 이 명령은 기존 `.env`를 덮어쓰지 않습니다. 명시한 port, device, model 값은 이번 실행에서 상속된 환경 값보다 우선합니다. 호스트 listener는 loopback에만 바인딩되지만 인증 gateway는 아닙니다. 공유하기 전에 [security guide(영문)](../examples/openai_api/SECURITY.md)를 확인하세요.

```bash
FUNASR_HOST_PORT=127.0.0.1:8000 FUNASR_DEVICE=cpu FUNASR_MODEL=sensevoice \
  docker compose -f examples/openai_api/docker-compose.yml up --build
```

Compose는 해당 터미널에서 계속 실행합니다. **두 번째 터미널을 열고 같은 저장소 루트**에서 Python 3.10 이상으로 확인하세요. smoke client는 표준 라이브러리만 사용하므로 호스트에 FunASR를 설치할 필요가 없습니다.

```bash
python3 examples/openai_api/smoke_test.py --base-url http://127.0.0.1:8000 --model sensevoice --response-format verbose_json
```

현재 디렉터리에 `sample.wav`가 없을 때만 공개 중국어 음성을 다운로드하며, 기존 파일이 있으면 재사용합니다. health, model metadata, 전사 JSON을 출력합니다. 텍스트를 직접 확인하세요. 종료 코드가 성공해도 인식 품질이나 동시 처리 성능을 검증한 것은 아닙니다. client는 Authorization을 보내지 않으며, security guide의 전사 전용 gateway는 이 smoke가 사용하는 metadata route를 거부합니다. 로컬 endpoint로 실행하고 민감한 음성이나 가리지 않은 출력을 남기지 마세요.

CPU image는 example server를 복사하지만 FunASR는 PyPI에서 설치합니다. 의존 버전은 고정되지 않으므로 재현성을 주장하기 전에 실제 package version과 image digest를 기록하세요. `FUNASR_DEVICE`만 변경해도 CUDA 의존성이나 container GPU access가 추가되지는 않습니다. GPU image와 scheduling은 별도로 준비하고 검증해야 합니다. [HTTP deployment guide(영문)](../examples/openai_api/README.md#docker-deployment)와 선택한 모델의 요구 사항을 확인하세요.

### Streaming 또는 live captioning이 필요하다

Runtime WebSocket service를 사용하세요. production 전에 chunk size, VAD, endpointing, punctuation, speaker diarization, reconnect, client backpressure를 실제 오디오로 검증하세요.

## Readiness checklist

- model alias를 정하고 deployment note에 고정합니다.
- FunASR version, model version, device, CUDA/PyTorch version, Docker image tag, command line을 기록합니다.
- public smoke sample과 realistic private sample을 최소 1개씩 실행합니다.
- request마다 audio duration, model, device, latency, response format, error type을 로깅합니다.
- trusted network 밖으로 API를 노출하기 전에 upload-size limit, authentication, TLS, rate limit을 넣습니다. [Security guide](../examples/openai_api/SECURITY.md)도 확인하세요.
- 막히면 deployment path, command/config, logs, model, device, audio characteristics를 포함해 [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md)를 열어 주세요.
