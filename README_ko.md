([English](./README.md)|[简体中文](./README_zh.md)|[日本語](./README_ja.md)|한국어)

<p align="center">
<a href="https://github.com/modelscope/FunASR"><img src="https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210" alt="FunASR"></a>
</p>

<p align="center">
  <strong>오프라인, 스트리밍 및 엣지 배포를 위한 산업용 음성 인식 툴킷.</strong><br>
  <em>ASR · VAD · 구두점 · 화자 파이프라인 · 감정 및 오디오 이벤트 모델 · OpenAI 호환 서빙</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/v/funasr" alt="PyPI"></a>
  <a href="https://github.com/modelscope/FunASR"><img src="https://img.shields.io/github/stars/modelscope/FunASR?style=social" alt="Stars"></a>
  <a href="https://pypi.org/project/funasr/"><img src="https://img.shields.io/pypi/dm/funasr" alt="Downloads"></a>
  <a href="https://modelscope.github.io/FunASR/"><img src="https://img.shields.io/badge/문서-온라인-blue" alt="Docs"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/10479" target="_blank"><img src="https://trendshift.io/api/badge/repositories/10479" alt="modelscope%2FFunASR | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
  <a href="#빠른-시작">빠른 시작</a> · <a href="./docs/model_selection_ko.md">모델 선택</a> · <a href="#모델-목록">모델 목록</a> · <a href="./docs/deployment_matrix_ko.md">배포 방식</a> · <a href="https://www.funasr.com/en/">배포 센터</a> · <a href="https://www.funasr.com/en/docs/">문서</a> · <a href="#벤치마크">벤치마크</a>
</p>

---

<a name="빠른-시작"></a>

## 빠른 시작

```bash
python -m pip install torch torchaudio
python -m pip install funasr
```

아래는 공개 샘플을 사용하는 CPU 우선 예제입니다. GPU를 사용하려면
[설치 가이드](./docs/installation/installation.md)에 따라 호환되는 PyTorch/CUDA
환경을 준비하고 `torch.cuda.is_available()`을 확인한 뒤 `device="cuda"`로 바꾸세요.

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cpu")
result = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")

for seg in result[0]["sentence_info"]:
    print(f"[{seg['start']/1000:.1f}s] 화자{seg['spk']}: {rich_transcription_postprocess(seg['sentence'])}")
```

실제로 반환된 VAD 구간의 시작 시각(초), 익명 화자 번호, SenseVoice 태그를 제거한
텍스트를 출력합니다. 텍스트와 구간은 오디오와 checkpoint에 따라 달라지므로
고정된 인식 결과를 제시하지 않습니다.

CAM++는 `spk_embedding` 벡터를 추출하고, `AutoModel`이 클러스터링하여 VAD
구간에 화자 번호를 할당합니다. 번호는 해당 녹음 안에서만 유효하며 알려진 인물의
신원이나 SenseVoiceSmall 단독 출력이 아닙니다. 자세한 내용은
[SDK 계약](./docs/python_api.md)을 참고하세요.

처음 사용한다면 [Colab 빠른 시작](./examples/colab/README_ko.md)으로 먼저 확인할 수 있습니다. 어떤 모델을 선택할지 고민된다면 [모델 선택 가이드](./docs/model_selection_ko.md)를 참고하세요.

> **API 서버로 배포:** [로컬 SenseVoice CPU 절차](#배포) · [Nano GPU 서빙과 고정 버전 vLLM 환경](./docs/vllm_guide.md)
>
> **AI Agent 연동:** [MCP 서버](examples/mcp_server/) Claude/Cursor 지원 · [OpenAI API](examples/openai_api/) LangChain/Dify/AutoGen 지원

### 왜 FunASR인가?

FunASR는 툴킷입니다. 작업, checkpoint, 런타임을 각각 선택해야 합니다.
한 모델이나 어댑터가 지원하는 기능이 모든 서빙 백엔드에서 지원되는 것은 아닙니다.

| 작업 | Checkpoint 또는 파이프라인 | 런타임 진입점 | 주요 제한 |
|---|---|---|---|
| 파일 전사와 감정/이벤트 태그 | SenseVoiceSmall | Python `AutoModel`, CPU 또는 GPU | 5개 언어 checkpoint이며 태그는 화자 신원을 나타내지 않습니다. |
| LLM 기반 파일 전사 | Fun-ASR-Nano | `AutoModel`, 또는 문서의 GPU 분리 엔진 `AutoModelVLLM` | 기본 Nano는 중/영/일 및 중국어 방언/억양을 지원하며 timestamp는 checkpoint와 경로에 따라 다릅니다. |
| 더 많은 언어의 파일 전사 | Fun-ASR-MLT-Nano | Python `AutoModel` | 별도의 31개 언어 checkpoint이며 기본 Nano에 같은 범위를 적용하지 않습니다. |
| 청크 단위 실시간 전사 | Paraformer-zh-streaming | 스트리밍 SDK 또는 runtime WebSocket | 스트리밍 checkpoint와 세션별 cache가 필요하며 오프라인 checkpoint로 대체할 수 없습니다. |
| 화자별 파일 전사 | SenseVoiceSmall + FSMN-VAD + CAM++ | `AutoModel`의 VAD와 임베딩 클러스터링 | 녹음 안의 익명 번호이며 등록된 인물의 신원 식별이 아닙니다. |
| 텍스트, 시각, 화자 공동 생성 | 제3자 OpenMOSS의 MOSS-Transcribe-Diarize | MOSS 가이드의 FunASR adapter 또는 업스트림 백엔드 | 오프라인, 녹음 내 익명 라벨이며 통합 경로에 외부 VAD/화자 모델을 붙이지 않습니다. |
| 네이티브 CPU/엣지 전사 | Fun-ASR-Nano 또는 SenseVoiceSmall GGUF | llama.cpp runtime | 호환되는 변환 가중치가 필요하며 GGUF는 Python `AutoModel`용 checkpoint가 아닙니다. |

[Model Zoo](./model_zoo/readme.md)와 [배포 매트릭스](./docs/deployment_matrix_ko.md)에서
인터페이스와 라이선스 제한을 확인하고 대상 오디오와 하드웨어로 평가하세요.

---

<a name="벤치마크"></a>

## 벤치마크

[기존 평가 보고서](https://modelscope.github.io/FunASR/benchmark.html)와
[분리 엔진 측정](./docs/vllm_guide.md#benchmark)에 원래 결과를 보존합니다.
서로 다른 기록이며 보편적인 속도 순위나 운영 용량을 보장하지 않습니다.

[RTFx와 재현성 설명](./docs/benchmark/rtf_reproducibility.md)에 따라
checkpoint/revision, 오디오 집합, 하드웨어, 배치, 워밍업, 측정 범위, CER/WER를
맞춰 비교하세요. 오프라인 처리량은 스트리밍 지연이 아닙니다.
[마이그레이션 평가 예제](./examples/migration/)로 자신의 녹음을 측정할 수 있습니다.

---

## 최신 소식

- **MOSS-Transcribe-Diarize**를 FunASR service, Docker, Kubernetes, vLLM/SGLang workflow, FunClip에 통합해 긴 오디오 ASR, timestamp, 익명 speaker label을 한 번에 처리합니다. [MOSS 배포 ->](./docs/moss_transcribe_diarize.md)
- **FunASR 1.4.15**는 테스트를 거친 NumPy 2 호환성을 추가하고 스트리밍 KWS/VAD 경계 처리와 checkpoint 순위 산정을 수정합니다. `python -m pip install -U "funasr==1.4.15"`. [릴리스 및 검증 범위 ->](https://github.com/modelscope/FunASR/releases/tag/v1.4.15)
- **Production deployment**에 더 빠르고 안정적인 realtime serving과 Linux, macOS, Windows 10개 target용 llama.cpp package를 추가했습니다. [GPU service ->](./docs/vllm_guide.md) · [CPU / edge package ->](https://www.funasr.com/en/deploy/llama-cpp.html) · [v0.2.6 binaries ->](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6)

> 전체 변경 기록과 download asset은 [GitHub Releases](https://github.com/modelscope/FunASR/releases)에서 확인할 수 있습니다.

---

## 설치

```bash
pip install funasr
```

요구사항: Python ≥ 3.8, PyTorch ≥ 1.13, torchaudio

---

<a name="모델-목록"></a>

## 모델 목록

제3자 모델도 포함합니다. MOSS-Transcribe-Diarize의 배포 주체는 **OpenMOSS**이며
FunASR는 어댑터를 제공합니다. 통합 경로는 오프라인이고 익명 화자 라벨은 해당 녹음
안에서만 유효합니다. 실시간 처리나 알려진 인물의 신원 식별이 아닙니다.
모델 라이선스는 툴킷의 MIT 라이선스와 별도로 확인해야 합니다.

| 모델 | 작업 | 언어 | 파라미터 | 링크 |
|------|------|------|---------|------|
| **Fun-ASR-Nano** | 인식 | 중/영/일 + 중국어 방언 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) [GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) |
| **Fun-ASR-MLT-Nano** | 인식 | 31개 언어 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) |
| **SenseVoiceSmall** | 인식 + 감정 + 이벤트 | 중/영/일/한/광둥어 | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) [GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) |
| **MOSS-Transcribe-Diarize** | 제3자 OpenMOSS: 오프라인 인식 + 타임스탬프 + 익명 화자 | 공식 모델 카드 참조 | 공식 모델 카드 참조 | [🤗](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) [가이드](./docs/moss_transcribe_diarize.md) |
| **Paraformer-zh** | 인식 + 타임스탬프 | 중/영 | 220M | [⭐](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Qwen3-ASR | 인식, 52개 언어 | 다국어 | 1.7B | [사용법](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | 인식, 17개 언어 | 다국어 | 1.5B | [사용법](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3-turbo | 인식 + 번역 | 다국어 | 809M | [사용법](examples/industrial_data_pretraining/whisper) |

---

## 배포

새 디렉터리에서 POSIX shell과 Python 3.11로 로컬 SenseVoice CPU 서비스를 시작합니다.
현재 소스 checkout이 아니라 PyPI 릴리스를 별도 환경에 설치합니다. 인증을 내장하지
않으므로 loopback에서만 수신하고, 다른 클라이언트에 공개하기 전에
[보안 가이드](./examples/openai_api/SECURITY.md)를 확인하세요.

```bash
python3.11 -m venv .venv-funasr-http
. .venv-funasr-http/bin/activate
python -m pip install torch torchaudio
python -m pip install funasr fastapi uvicorn python-multipart
python -m pip check
funasr-server --host 127.0.0.1 --port 8000 --model sensevoice --device cpu
```

모델 다운로드와 서버 시작을 기다리세요. 두 번째 터미널에서 같은 디렉터리로 이동한 뒤
curl 7.76+로 공개 중국어 샘플을 내려받아 전사합니다. 요청은 미리 로드한 모델과
일치하며, 고정된 텍스트나 화자 라벨을 보장하지 않습니다.

```bash
curl --fail --location https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav && \
curl --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

오프라인 ASR과 익명 화자 라벨을 함께 생성하려면 (`moss-transcribe-diarize`)
[MOSS service / Docker / Kubernetes / vLLM / SGLang / LocalAI / FunClip guide →](./docs/moss_transcribe_diarize.md)의
별도 환경을 준비하세요. 위 CPU 환경에서 이어서 실행하는 명령이 아니라 대체 서비스입니다.
8000 포트를 다시 쓰기 전에 CPU 서비스를 중지하세요. Nano GPU 서빙은
[고정 버전 분리 엔진 가이드](./docs/vllm_guide.md)를 따르고 실제 backend 로그를 확인하세요.
모델 선택만으로 vLLM이 로드되었다고 판단할 수는 없습니다.

```bash
# Docker 스트리밍 서비스
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
```

CPU/엣지에서 Python 없이 오프라인 ASR만 필요하다면 llama.cpp / GGUF 런타임을 사용할 수 있습니다: [funasr.com/deploy/llama-cpp](https://www.funasr.com/en/deploy/llama-cpp.html) · [Fun-ASR-Nano-GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) · [SenseVoiceSmall-GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF).

**사전 빌드 바이너리:** [Releases](https://github.com/modelscope/FunASR/releases) · [v0.2.6](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6) · [Linux Vulkan tarball](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-linux-x64-vulkan.tar.gz) · [Windows Vulkan zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-vulkan.zip) · [Windows CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda.zip) · [Windows Blackwell CUDA zip](https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.6/funasr-llamacpp-windows-x64-cuda-blackwell.zip) · **다운로드와 빠른 시작:** [funasr.com/deploy/llama-cpp](https://www.funasr.com/en/deploy/llama-cpp.html) · **GGUF 모델:** [Hugging Face](https://huggingface.co/FunAudioLLM) · **문서와 벤치마크:** [runtime/llama.cpp/](./runtime/llama.cpp/)

Windows GPU에서는 [runtime-llamacpp-v0.2.6](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.6)의 `windows-x64-vulkan`, `windows-x64-cuda` 또는 `windows-x64-cuda-blackwell` 패키지를 선택하세요. RTX 50 / Blackwell (`sm_120`)에는 전용 `windows-x64-cuda-blackwell` 패키지가 있습니다. CI 아카이브 검증은 실제 Blackwell 하드웨어에서의 추론을 보장하지 않습니다. 자세한 내용은 [llama.cpp 배포 가이드](https://www.funasr.com/en/deploy/llama-cpp.html)를 참조하세요.

[Colab quickstart →](./examples/colab/README_ko.md) · [OpenAI API example →](./examples/openai_api/README_ko.md) · [Client recipes →](./examples/openai_api/CLIENTS.md) · [Workflow recipes →](./examples/openai_api/WORKFLOWS.md) · [Postman collection →](./examples/openai_api/POSTMAN.md) · [OpenAPI spec →](./examples/openai_api/OPENAPI.md) · [Security guide →](./examples/openai_api/SECURITY.md) · [Deployment matrix →](./docs/deployment_matrix_ko.md) · [배포 문서 →](./runtime/readme.md) · [Agent 연동 →](https://modelscope.github.io/FunASR/agent.html)

---

## 커뮤니티

|  |  |
|---|---|
| 📖 [문서](https://modelscope.github.io/FunASR/) | 🐛 [Issues](https://github.com/modelscope/FunASR/issues) |
| 💬 [Discussions](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |

## 라이선스

- 이 저장소의 FunASR 툴킷 소스 코드: [MIT License](./LICENSE).
- 사전 학습된 모델 가중치는 별도로 라이선스됩니다. 각 모델 카드에 표시된 라이선스를 확인하세요. 모델 카드가 이 저장소의 [FunASR Model Open Source License Agreement](./MODEL_LICENSE)를 가리키는 경우 해당 조건이 적용됩니다.
