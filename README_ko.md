([English](./README.md)|[简体中文](./README_zh.md)|[日本語](./README_ja.md)|한국어)

<p align="center">
<a href="https://github.com/modelscope/FunASR"><img src="https://svg-banners.vercel.app/api?type=origin&text1=FunASR🤠&text2=💖%20A%20Fundamental%20End-to-End%20Speech%20Recognition%20Toolkit&width=800&height=210" alt="FunASR"></a>
</p>

<p align="center">
  <strong>산업용 음성인식. 최대 340배 실시간, Whisper보다 26배 빠름. 50개 이상 언어 지원.</strong><br>
  <em>화자 분리 · 감정 인식 · 스트리밍 · 한 번의 호출로 해결</em>
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

하나의 모델, 한 번의 호출 — VAD 분할, 음성인식, 구두점 복원, 화자 분리가 모두 자동으로 수행됩니다.

처음 사용한다면 [Colab 빠른 시작](./examples/colab/README_ko.md)으로 먼저 확인할 수 있습니다. 어떤 모델을 선택할지 고민된다면 [모델 선택 가이드](./docs/model_selection_ko.md)를 참고하세요.

> **API 서버로 배포:** `funasr-server --device cuda` → localhost:8000에서 OpenAI 호환 엔드포인트
>
> **AI Agent 연동:** [MCP 서버](examples/mcp_server/) Claude/Cursor 지원 · [OpenAI API](examples/openai_api/) LangChain/Dify/AutoGen 지원

### 왜 FunASR인가?

Whisper는 단일 모델이지만, **FunASR는 툴킷**입니다 — 용도에 맞는 모델을 고르세요: **Fun-ASR-Nano**(플래그십 LLM-ASR, GPU 필요, vLLM로 340배 실시간, 31개 언어), **SenseVoice**(CPU 친화적, 감정·오디오 이벤트 포함), **Paraformer**(저지연 스트리밍). 아래 표는 단일 Whisper 모델 대비 툴킷이 제공하는 것이며, 각 기능에는 이를 제공하는 모델을 표기했습니다:

| | FunASR(툴킷) | Whisper | 클라우드 API |
|---|---|---|---|
| 최고 속도 | **340배 실시간**(Fun-ASR-Nano + vLLM) | 13배 실시간 | ~1배 실시간 |
| 화자 인식 | ✅ 내장 | ❌ pyannote 필요 | ✅ 추가 비용 |
| 감정 인식 | ✅ SenseVoice 제공 | ❌ | ❌ |
| 언어 수 | 50개 이상(Qwen3-ASR 52, Nano 31) | 57개 | 서비스마다 다름 |
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

- 2026/05/24: **v1.3.3** — `funasr-server` CLI, OpenAI 호환 API, MCP 서버. `pip install --upgrade funasr`
- 2026/05/20: Qwen3-ASR (0.6B/1.7B) 추가 — 52개 언어 지원.
- 2026/05/20: GLM-ASR-Nano (1.5B) 추가 — 17개 언어, 방언 지원.
- 2025/12/15: [Fun-ASR-Nano-2512](https://github.com/FunAudioLLM/Fun-ASR) — 31개 언어 지원.

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
| **Fun-ASR-Nano** | 인식 + 타임스탬프 | 31개 언어 | 800M | [⭐](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) [🤗](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) |
| **SenseVoiceSmall** | 인식 + 감정 + 이벤트 | 중/영/일/한/광둥어 | 234M | [⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall) [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| **Paraformer-zh** | 인식 + 타임스탬프 | 중/영 | 220M | [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) [🤗](https://huggingface.co/funasr/paraformer-zh) |
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

[Colab quickstart →](./examples/colab/README_ko.md) · [OpenAI API example →](./examples/openai_api/README_ko.md) · [Client recipes →](./examples/openai_api/CLIENTS.md) · [Workflow recipes →](./examples/openai_api/WORKFLOWS.md) · [Postman collection →](./examples/openai_api/POSTMAN.md) · [OpenAPI spec →](./examples/openai_api/OPENAPI.md) · [Security guide →](./examples/openai_api/SECURITY.md) · [Deployment matrix →](./docs/deployment_matrix_ko.md) · [배포 문서 →](./runtime/readme.md) · [Agent 연동 →](https://modelscope.github.io/FunASR/agent.html)

---

## 커뮤니티

|  |  |
|---|---|
| 📖 [문서](https://modelscope.github.io/FunASR/) | 🐛 [Issues](https://github.com/modelscope/FunASR/issues) |
| 💬 [Discussions](https://github.com/modelscope/FunASR/discussions) | 🤗 [HuggingFace](https://huggingface.co/funasr) |

## 라이선스

- 이 저장소의 FunASR 툴킷 소스 코드: [MIT License](./LICENSE)
- 사전 학습된 모델 가중치는 별도로 라이선스됩니다. 각 모델 카드에 표시된 라이선스를 확인하세요. 모델 카드가 이 저장소의 [FunASR Model Open Source License Agreement](./MODEL_LICENSE)를 가리키는 경우 해당 조건이 적용됩니다.
