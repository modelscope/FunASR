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

데모, 프라이빗 API, 다국어 전사, Agent 음성 입력을 여기서 평가할 수 있습니다. 위 회의록 예제는 SenseVoice의 ASR 및 감정/이벤트 태그와 별도로, `fsmn-vad`로 음성 구간을 찾고 `cam++`의 화자 임베딩을 클러스터링합니다. 화자 라벨은 녹음 내 익명 번호이며, 등록된 인물의 식별이나 녹음 간 고정 ID가 아닙니다. 대상 언어와 실제 오디오로 모델을 비교하세요.

**Fun-ASR-Nano-2512**의 중국어·영어·일본어 및 중국어 방언/지역 억양 지원이 한국어 지원을 뜻하지는 않습니다. **Fun-ASR-MLT-Nano**는 별도 checkpoint입니다. 한국어를 포함한 대상 언어의 지원 범위는 각 모델 카드를 확인하고, Nano의 범위와 혼동하지 마세요.

## 결정 표

| 필요 | 먼저 시도할 것 | 이유 | 다음 문서 |
|---|---|---|---|
| 빠른 다국어 프라이빗 전사 | SenseVoice-Small | ASR, 감정 태그, 음성 이벤트 태그, CPU/GPU 사용성이 균형 잡힌 기본 경로입니다. | [README quick start](../README_ko.md#빠른-시작) |
| 중국어 중심 프로덕션 ASR | Paraformer-Large | VAD와 문장부호 복원을 함께 쓰는 성숙한 중국어 ASR 경로입니다. | [Tutorial](./tutorial/README.md) |
| OpenAI API 예제의 영어 경로 | `paraformer-en` alias | OpenAI-style client에서 호환성을 확인하기 쉬운 가벼운 영어 경로입니다. | [OpenAI API example](../examples/openai_api/README_ko.md) |
| LLM-based ASR 또는 중영일 + 중국어 방언/지역 억양 평가 | Fun-ASR-Nano | Python에서 평가한 뒤 checkpoint와 interface에 맞는 vLLM 경로를 선택합니다. | [vLLM 경로 선택](#vllm-checkpoint-paths) |
| 오프라인 장시간 ASR 및 익명 화자 라벨 | MOSS-Transcribe-Diarize | 한 번의 오프라인 request로 전사, timestamps, 녹음 내 익명 화자 라벨을 반환합니다. 알려진 인물을 식별하지 않으며 외부 VAD / speaker model도 필요하지 않습니다. | [MOSS deployment guide](./moss_transcribe_diarize.md) |
| 라이브 자막 또는 콜센터 스트림 | Runtime WebSocket service | 장시간 연결, 부분 결과, endpointing에 맞춘 런타임입니다. | [Runtime service docs](../runtime/readme.md) |
| Whisper / cloud ASR에서 전환 | SenseVoice-Small로 baseline을 만들고 필요하면 비교 | 강한 기본 경로로 먼저 평가한 뒤 용도별로 조정하는 편이 안전합니다. | [Migration guide](./migration_from_whisper.md) |

## OpenAI 호환 API alias

`examples/openai_api` server는 짧은 alias를 제공합니다. 애플리케이션 팀은 모델 repository ID를 몰라도 사용할 수 있습니다.

- **`sensevoice`**: `iic/SenseVoiceSmall`을 사용하는 CPU/GPU 다국어 HTTP 전사입니다. 반환 텍스트에서 리치 태그는 제거됩니다.
- **`paraformer`**: `paraformer-zh`에 VAD와 문장부호 복원을 결합한 중국어 경로입니다.
- **`paraformer-en`**: `paraformer-en`과 VAD를 사용하는 OpenAI-style client용 영어 전사입니다.
- **`fun-asr-nano`**: `FunAudioLLM/Fun-ASR-Nano-2512`로 중영일·중국어 방언/지역 억양을 평가합니다. vLLM acceleration을 시험할 때는 호환되는 runtime을 선택하세요.
- **`moss-transcribe-diarize`**: 서드파티 `OpenMOSS-Team/MOSS-Transcribe-Diarize`의 오프라인 전사와 녹음 내 익명 화자 라벨입니다. [MOSS guide(영문)](./moss_transcribe_diarize.md)에서 전용 의존성과 remote code를 검토하고, 구조화된 segment에는 `verbose_json`을 요청하세요. 외부 VAD / speaker model이 필요 없으며 알려진 인물을 식별하지 않습니다.

여기서 설명하는 alias는 `AutoModel`을 로드하는 [예제 server](../examples/openai_api/server.py)의 설정입니다.
native vLLM이나 `AutoModelVLLM`을 자동으로 선택하지 않습니다.
패키지의 `funasr-server`는 별도 loader / backend 선택 로직을 사용하므로,
서비스 사이에서 alias나 성능 결과를 그대로 재사용하지 마세요.

이 HTTP 예제는 최상위 `text`와 `verbose_json`의 각 segment `text`를 정리하므로,
형식을 바꿔도 감정/이벤트 태그가 복원되지 않습니다. 원래 태그가 필요하면 Python SDK를
사용하고 표시용 후처리 전에 반환된 `text`를 보존하세요.
[원본 태그 보존 레시피(영문)](./speaker_emotion.md)를 참고하세요.

클라이언트를 연결하기 전에 서비스를 확인하세요.

```bash
curl http://localhost:8000/v1/models
python examples/openai_api/smoke_test.py --base-url http://localhost:8000 --model sensevoice
```

SDK, JavaScript, workflow, Postman, OpenAPI, Docker, Kubernetes는 [OpenAI API example](../examples/openai_api/README_ko.md)에서 시작하세요.

<a id="vllm-checkpoint-paths"></a>

## vLLM checkpoint와 interface 선택

| 경로 | checkpoint와 interface | 다음 문서 |
| --- | --- | --- |
| FunASR split-engine | 기본 `FunAudioLLM/Fun-ASR-Nano-2512`를 `AutoModelVLLM`으로 로드합니다. FunASR은 오디오 부분, vLLM은 decoder를 처리합니다. | [Split-engine(영문)](./vllm_guide.md) |
| 공식 native vLLM | 변환된 `FunAudioLLM/Fun-ASR-Nano-2512-vllm`을 vLLM의 native 구현으로 로드하고 `/v1/audio/transcriptions`를 사용합니다. `AutoModelVLLM` 로드가 아닙니다. | [공식 기능 검증(영문)](./vllm_official_native_validation.md) |
| 과거 community native vLLM | `allendou/Fun-ASR-Nano-2512-vllm`, 2026-08-13 검증입니다. 측정 시간은 당시 checkpoint와 환경에만 해당합니다. | [과거 community 기록](./vllm_native_funasr_validation.md) |

공식 기록은 고정 revision과 기존 환경에서의 기능 검증이며, 신규 설치 절차,
지속 부하 benchmark 또는 `/v1/realtime` streaming 검증이 아닙니다.
과거 community 측정 시간을 공식 모델의 결과로 사용하지 마세요.
MOSS는 별도 가이드를 따르세요. Nano checkpoint와 검증은 MOSS 호환성을 입증하지 않습니다.

## 벤치마크 후 결정하기

깨끗한 demo 오디오 하나만 보고 모델을 정하지 마세요. 먼저 작은 대표 세트로 확인합니다.

- 짧은 클립, 긴 회의, 무음, 잡음, 화자 겹침, 도메인 용어, 대상 언어를 포함하는 20-50개 파일을 준비합니다.
- model name, model revision, FunASR version, device, CPU/GPU, CUDA/PyTorch, runtime path, batch size, download/warmup 처리 여부를 기록합니다.
- 읽기 쉬움만 보지 말고, 평소 사용하는 WER/CER 또는 사람 리뷰로 품질을 확인합니다.
- latency, throughput, memory, failure, upload size limit을 함께 비교합니다.
- 막히면 model, device, command, logs, audio duration, runtime path를 포함해 [Deployment Help issue](https://github.com/modelscope/FunASR/issues/new?template=deployment_help.md)를 열어 주세요.
