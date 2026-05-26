# FunASR 모델 선택 가이드

처음 FunASR을 사용할 때, Whisper나 클라우드 ASR에서 전환할 때, 또는 OpenAI 호환 API에서 노출할 model alias를 정할 때 참고하세요.

## 고민된다면 여기서 시작

먼저 **SenseVoice-Small**을 추천합니다.

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    spk_model="cam++",
    device="cuda",  # 간단한 smoke test는 "cpu"도 가능
)
result = model.generate(input="meeting.wav")
```

데모, 프라이빗 API, 다국어 전사, 화자 포함 회의록, Agent 음성 입력의 첫 선택지로 사용하기 좋습니다. 중국어 프로덕션 정확도, 스트리밍 지연 시간, LLM-based ASR 평가처럼 명확한 요구가 있을 때만 다른 경로로 전환하세요.

## 결정 표

| 필요 | 먼저 시도할 것 | 이유 | 다음 문서 |
|---|---|---|---|
| 빠른 다국어 프라이빗 전사 | SenseVoice-Small | ASR, 감정 태그, 음성 이벤트 태그, CPU/GPU 사용성이 균형 잡힌 기본 경로입니다. | [README quick start](../README_ko.md#빠른-시작) |
| 중국어 중심 프로덕션 ASR | Paraformer-Large | VAD와 문장부호 복원을 함께 쓰는 성숙한 중국어 ASR 경로입니다. | [Tutorial](./tutorial/README.md) |
| OpenAI API 예제의 영어 경로 | `paraformer-en` alias | OpenAI-style client에서 호환성을 확인하기 쉬운 가벼운 영어 경로입니다. | [OpenAI API example](../examples/openai_api/README_ko.md) |
| LLM-based ASR 또는 31개 언어 평가 | Fun-ASR-Nano | LLM-based 모델입니다. decoder throughput이 중요하면 vLLM을 사용합니다. | [vLLM guide](./vllm_guide.md) |
| 라이브 자막 또는 콜센터 스트림 | Runtime WebSocket service | 장시간 연결, 부분 결과, endpointing에 맞춘 런타임입니다. | [Runtime service docs](../runtime/readme.md) |
| Whisper / cloud ASR에서 전환 | SenseVoice-Small로 baseline을 만들고 필요하면 비교 | 강한 기본 경로로 먼저 평가한 뒤 용도별로 조정하는 편이 안전합니다. | [Migration guide](./migration_from_whisper.md) |

## OpenAI 호환 API alias

`examples/openai_api` server는 짧은 alias를 제공합니다. 애플리케이션 팀은 모델 repository ID를 몰라도 사용할 수 있습니다.

| Alias | 내부 경로 | 사용 시점 |
|---|---|---|
| `sensevoice` | `iic/SenseVoiceSmall` | 다국어 ASR, 이벤트 태그, CPU/GPU 동작이 균형 잡힌 기본 프라이빗 음성 API. |
| `paraformer` | `paraformer-zh` + VAD + punctuation | 중국어 중심 프로덕션 경로. |
| `paraformer-en` | `paraformer-en` + VAD | OpenAI-style client의 영어 호환성 확인. |
| `fun-asr-nano` | `FunAudioLLM/Fun-ASR-Nano-2512` | LLM-based ASR, 31개 언어, vLLM acceleration 평가. |

클라이언트를 연결하기 전에 서비스를 확인하세요.

```bash
curl http://localhost:8000/v1/models
python examples/openai_api/smoke_test.py --base-url http://localhost:8000 --model sensevoice
```

SDK, JavaScript, workflow, Postman, OpenAPI, Docker, Kubernetes는 [OpenAI API example](../examples/openai_api/README_ko.md)에서 시작하세요.

## 벤치마크 후 결정하기

깨끗한 demo 오디오 하나만 보고 모델을 정하지 마세요. 먼저 작은 대표 세트로 확인합니다.

- 짧은 클립, 긴 회의, 무음, 잡음, 화자 겹침, 도메인 용어, 대상 언어를 포함하는 20-50개 파일을 준비합니다.
- model name, model revision, FunASR version, device, CPU/GPU, CUDA/PyTorch, runtime path, batch size, download/warmup 처리 여부를 기록합니다.
- 읽기 쉬움만 보지 말고, 평소 사용하는 WER/CER 또는 사람 리뷰로 품질을 확인합니다.
- latency, throughput, memory, failure, upload size limit을 함께 비교합니다.
- 막히면 model, device, command, logs, audio duration, runtime path를 포함해 [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md)를 열어 주세요.
