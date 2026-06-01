# FunASR Project Updates & Strategic Assessment (June 2026)

This document tracks the evolution of FunASR and provides a "warts and all" assessment of its current stability for integration into other projects.

## 🚨 Repository Alert Status (Updated June 1, 2026)
**Current Status:** **IMPROVED STABILITY** on `main` branch (Version 1.3.9).

### 1. Recent Major Updates (June 2026)
- **CPU Support (Production Path):** Realtime WebSocket server now includes a robust bypass for vLLM when running on non-CUDA devices (CPU), utilizing standard `AutoModel` (PyTorch) with improved argument handling.
- **Production Infrastructure:** Massive expansion of `openai_api` examples, including Gradio, Kubernetes manifests, and full API specification for easier enterprise integration.
- **Improved Performance:** Refactored streaming buffer management to resolve O(N^2) complexity issues.
- **Model Support:** Enhanced native support for Whisper and GLM models.

### 2. The Transformers v5 "Deadlock" (Still Present)
- **The Problem:** Modern stacks (Transformers 5.x) are incompatible with `qwen-asr` (v0.0.6) and other sub-models. It triggers an `AttributeError: 'Qwen3ASRConfig' object has no attribute 'thinker_config'`.
- **The Workaround:** Users **must** pin their environment: `pip install "transformers==4.57.6" "huggingface-hub<1.0"`.

---

## 🧪 Experimental Results (Polish ASR)

### 1. Model Quality Tier List (Polish)

| Tier | Model | Parameters | Verdict |
| :--- | :--- | :--- | :--- |
| **Godzilla** | **`Qwen/Qwen2-Audio-7B-Instruct`** | **7 Billion** | **Peak Intelligence.** Best accuracy for Polish. Extremely heavy on CPU. |
| **Industrial** | **`FunAudioLLM/Fun-ASR-MLT-Nano-2512`** | **800 Million** | **The Workhorse.** Optimized for 31 languages. Balanced quality/speed. |
| **Mobile** | `iic/SenseVoiceSmall` | 234 Million | **Fast Default.** Great for low-latency tasks. CPU viable. |

---

## ⚖️ Strategic Comparison: FunASR vs. WhisperX

| Metric | WhisperX (The Scribe) | FunASR (The Engine) |
| :--- | :--- | :--- |
| **CPU Performance** | 🥇 **Champion.** Optimized C++. | 🥉 **Laggard.** Unoptimized PyTorch. |
| **Polish Accuracy** | 🥇 **Stable.** | 🥈 **Variable.** Needs "Godzilla" models to compete. |
| **Orchestration** | 🥈 **Rigid.** | 🥇 **Powerful.** VAD + SPK + Emotion in one pipeline. |

**Verdict on Migration:** 
Use FunASR for **complex pipelines** (Diarization, Emotion, Hotwords) or **high-end GPUs**. Stick with WhisperX for simple, fast, CPU-bound Polish transcription.

---

## 🛠️ Recommended Components & Best Practices
1. **`funasr/fsmn-vad`**: Efficient segmentation.
2. **`funasr/campplus`**: Speaker identity.
3. **Configuration**: Use `hub="hf"`, `trust_remote_code=True`.
4. **vLLM Integration**: For production NVIDIA GPUs, use `AutoModelVLLM`. For CPU/General use, `AutoModel` is now supported via device detection bypass.
