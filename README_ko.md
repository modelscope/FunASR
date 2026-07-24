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
  <a href="#빠른-시작">빠른 시작</a> · <a href="./examples/colab/README_ko.md">Colab</a> · <a href="./docs/model_selection_ko.md">모델 선택</a> · <a href="#벤치마크">벤치마크</a> · <a href="./docs/migration_from_whisper.md">Migration guide</a> · <a href="./docs/use_case_showcase.md">Use cases</a> · <a href="./docs/deployment_matrix_ko.md">Deployment matrix</a> · <a href="#모델-목록">모델 목록</a> · <a href="https://modelscope.github.io/FunASR/agent.html">Agent 연동</a> · <a href="https://modelscope.github.io/FunASR/">문서</a>
</p>

---

<a name="빠른-시작"></a>

## 빠른 시작

```bash
pip install funasr
```

```python
from funasr import AutoModel

model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++", device="cuda")
result = model.generate(input="meeting.wav")
```

**출력** — 화자 라벨, 타임스탬프, 구두점이 포함된 구조화된 텍스트:
```
[00:00.4 → 00:03.8] 화자0: Q3 계획에 대해 논의하겠습니다.
[00:04.2 → 00:07.1] 화자1: 좋습니다. 세 가지 포인트가 있습니다.
[00:07.5 → 00:12.3] 화자0: 말씀하세요. 30분 남았습니다.
```

한 번의 `AutoModel` 파이프라인 호출이지만, SenseVoiceSmall, FSMN-VAD,
CAM++라는 독립된 모델을 조합합니다. 화자 분리는 SenseVoiceSmall 자체가 아니라
CAM++에서 제공합니다.

처음 사용한다면 [Colab 빠른 시작](./examples/colab/README_ko.md)으로 먼저 확인할 수 있습니다. 어떤 모델을 선택할지 고민된다면 [모델 선택 가이드](./docs/model_selection_ko.md)를 참고하세요.

> **API 서버로 배포:** `funasr-server --device cuda` → localhost:8000에서 OpenAI 호환 엔드포인트
>
> **AI Agent 연동:** [MCP 서버](examples/mcp_server/) Claude/Cursor 지원 · [OpenAI API](examples/openai_api/) LangChain/Dify/AutoGen 지원

### 왜 FunASR인가?

Whisper는 단일 모델이지만, **FunASR는 툴킷**입니다. 용도에 맞게
**Fun-ASR-Nano**(중국어/영어/일본어 및 중국어 방언/지역 억양, GPU),
**Fun-ASR-MLT-Nano**(31개 언어), **SenseVoiceSmall**(5개 언어 ASR와
감정·오디오 이벤트), **Paraformer**(저지연 스트리밍)를 선택하세요.
아래 표는 툴킷 전체의 기능과 이를 제공하는 모델 또는 파이프라인을 보여 줍니다:

| | FunASR(툴킷) | Whisper | 클라우드 API |
|---|---|---|---|
| 최고 속도 | **340배 실시간**(Fun-ASR-Nano + vLLM) | 13배 실시간 | ~1배 실시간 |
| 화자 인식 | ✅ VAD + CAM++ 파이프라인 | ❌ pyannote 필요 | ✅ 추가 비용 |
| 감정 인식 | ✅ SenseVoice 제공 | ❌ | ❌ |
| 언어 수 | 체크포인트별 상이(예: Qwen3-ASR 52, MLT-Nano 31, Nano 중/영/일) | 57개 | 서비스마다 다름 |
| 스트리밍 | ✅ WebSocket(Paraformer) | ❌ | ✅ |
| CPU 사용 | ✅ 17배 실시간(SenseVoice) | ❌ 너무 느림 | 해당 없음 |
| 자체 호스팅 | ✅ 지원 (툴킷: MIT, 모델별 상이) | ✅ MIT 라이선스 | ❌ 클라우드만 |
| 비용 | 무료 | 무료 | $0.006/분~ |

---

<a name="벤치마크"></a>

## 벤치마크

> 184개 장시간 오디오(총 192분). [상세 보고서 →](https://modelscope.github.io/FunASR/benchmark.html)

| 모델 | 중국어 CER ↓ | GPU 속도 | CPU 속도 | Whisper-large-v3 대비 |
|------|------|----------|----------|---------------------|
| **Fun-ASR-Nano**(vLLM) | **8.20%** | **340배** 실시간 | — | 🚀 **26배 빠름** |
| **SenseVoice-Small** | **7.81%** | **170배** 실시간 | **17배** 실시간 | 🚀 **13배 빠름** |
| **Paraformer-Large** | 10.18% | **120배** 실시간 | **15배** 실시간 | 🚀 **9배 빠름** |
| Whisper-large-v3-turbo | 21.71% | 46배 실시간 | ❌ | 3.4배 빠름 |
| Whisper-large-v3 | 20.02% | 13배 실시간 | ❌ | 기준선 |

> **핵심:** FunASR의 CPU 속도가 Whisper의 GPU 속도보다 빠릅니다.

---

## 최신 소식

- 2026/07/24: **v1.3.29 hotfix PyPI 공개** — SenseVoice 장시간 오디오 추론에서 word timestamp와 구두점 모델이 없을 때도 각 VAD 음성 구간을 `sentence_info`로 반환합니다. 자막 클라이언트는 길이가 0이거나 미디어 전체를 덮는 단일 cue 대신 인식 텍스트와 실제 밀리초 단위 시작·종료 시간을 받을 수 있습니다. 설치: `python -m pip install -U "funasr==1.3.29"`. [Release →](https://github.com/modelscope/FunASR/releases/tag/v1.3.29)
- 2026/07/24: **v1.3.28 hotfix PyPI 공개** — VAD로 확정된 realtime WebSocket 최종 결과가 짧은 접두사, 반복 hallucination 또는 decode 예외로 퇴화하면 현재 음성 구간을 연속해서 완전히 덮는 clean partial을 보존합니다. 짧은 STOP tail, VAD finalize, 화자 완료 처리도 동일한 안정적인 경로로 통합했습니다. SenseVoice 자막은 rich tag, 구두점, word/BPE timestamp를 올바르게 정렬해 중국어가 하나의 cue로 합쳐지거나 영어 원문이 손상되지 않습니다. 설치: `python -m pip install -U "funasr==1.3.28"`. [Release →](https://github.com/modelscope/FunASR/releases/tag/v1.3.28)
- 2026/07/24: **v1.3.27 PyPI 공개** — OpenAI 호환 서버가 `verbose_json`에 SenseVoice 감지 언어를 반환하고, vLLM fallback 후에는 캐시된 Fun-ASR-Nano `AutoModel`을 재사용합니다. vLLM/VAD 초기화와 fallback이 모두 실패하면 반쯤 초기화된 상태를 남기지 않아 이후 요청에서 다시 시도할 수 있습니다. 설치: `python -m pip install -U "funasr==1.3.27"`. [Release →](https://github.com/modelscope/FunASR/releases/tag/v1.3.27)
- 2026/07/23: **v1.3.26 PyPI 공개** — `funasr-server --model fun-asr-nano --hub ms`가 vLLM 경로와 AutoModel fallback 모두에서 ModelScope hub 선택을 존중합니다. 설치: `python -m pip install -U "funasr==1.3.26"`. [Release →](https://github.com/modelscope/FunASR/releases/tag/v1.3.26)
- 2026/07/23: **llama.cpp runtime v0.1.9** — Windows Vulkan용 `funasr-llamacpp-windows-x64-vulkan.zip`을 추가했습니다. 최신 AMD, Intel 또는 NVIDIA Vulkan 드라이버에서 SenseVoiceSmall을 독립 실행할 수 있습니다. Linux Vulkan, Windows CUDA, CPU/AVX2, Linux arm64, macOS arm64 패키지도 계속 제공합니다. [Release →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.9)
- 2026/07/22: **llama.cpp runtime v0.1.8** — Linux Vulkan tarball과 Windows CUDA zip을 포함한 CPU/엣지용 GGUF 런타임입니다. 다운로드와 빠른 시작: [funasr.com/llama-cpp](https://www.funasr.com/llama-cpp.html) · [Release →](https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.8)
- 2026/05/24: **v1.3.3** — `funasr-server` CLI, OpenAI 호환 API, MCP 서버. `pip install --upgrade funasr`
- 2026/05/20: Qwen3-ASR (0.6B/1.7B) 추가 — 52개 언어 지원.
- 2026/05/20: GLM-ASR-Nano (1.5B) 추가 — 17개 언어, 방언 지원.
- 2026/05/19: Fun-ASR-Nano 및 SenseVoice는 VAD 및 CAM++와 결합하여 화자 분리 파이프라인을 구성할 수 있습니다.
- 2025/12/15: [Fun-ASR-Nano-2512](https://github.com/QwenAudio/Fun-ASR) — 중국어, 영어, 일본어 및 중국어 방언 지원.

---

## 설치

```bash
pip install funasr
```

요구사항: Python ≥ 3.8, PyTorch ≥ 1.13, torchaudio

---

<a name="모델-목록"></a>

## 모델 목록

| 모델 | 작업 | 언어 | 파라미터 | 링크 |
|------|------|------|---------|------|
| **Fun-ASR-Nano** | 인식 | 중/영/일 + 중국어 방언 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) [GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) |
| **Fun-ASR-MLT-Nano** | 인식 | 31개 언어 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) |
| **SenseVoiceSmall** | 인식 + 감정 + 이벤트 | 중/영/일/한/광둥어 | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) [GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) |
| **Paraformer-zh** | 인식 + 타임스탬프 | 중/영 | 220M | [⭐](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
| Qwen3-ASR | 인식, 52개 언어 | 다국어 | 1.7B | [사용법](examples/industrial_data_pretraining/qwen3_asr) |
| GLM-ASR-Nano | 인식, 17개 언어 | 다국어 | 1.5B | [사용법](examples/industrial_data_pretraining/glm_asr) |
| Whisper-large-v3-turbo | 인식 + 번역 | 다국어 | 809M | [사용법](examples/industrial_data_pretraining/whisper) |

---

## 배포

```bash
# OpenAI 호환 API (권장)
pip install funasr fastapi uvicorn python-multipart
funasr-server --device cuda

# Docker 스트리밍 서비스
docker pull registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-online-cpu-0.1.12
```

CPU/엣지에서 Python 없이 오프라인 ASR만 필요하다면 llama.cpp / GGUF 런타임을 사용할 수 있습니다: [funasr.com/llama-cpp](https://www.funasr.com/llama-cpp.html) · [Fun-ASR-Nano-GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF) · [SenseVoiceSmall-GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF).

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
