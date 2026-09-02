# Offline long-form Qwen3-ASR with vLLM

`transcribe_vllm_offline.py` uses Qwen3-ASR's native `Qwen3ASRModel.LLM` backend. It does
not use `AutoModelVLLM`, which targets FunASR-native models and does not currently support
Qwen3-ASR.

The example normalizes input to mono 16 kHz audio, reuses qwen-asr's silence-aware audio
splitter, transcribes each chunk with native vLLM, and restores offsets on the source
timeline. The default 180-second limit matches qwen-asr's long-form boundary. No separate
VAD model is required for this path.

## Install

Use an isolated environment because `qwen-asr[vllm]==0.0.6` pins `vllm==0.14.0`:

```bash
python -m venv .venv-qwen3-vllm
source .venv-qwen3-vllm/bin/activate
pip install -U "qwen-asr[vllm]==0.0.6" "transformers==4.57.6"
```

An `ffmpeg` executable must also be available.

## Run

```bash
python examples/industrial_data_pretraining/qwen3_asr/transcribe_vllm_offline.py \
  recording.mp3 \
  --model Qwen/Qwen3-ASR-1.7B \
  --language Chinese \
  --max-inference-batch-size 4
```

The default output is `recording.qwen3-vllm.json`, containing the combined transcript and
each chunk's `start_ms`, `end_ms`, `text`, and detected language. Omit `--language` to
enable language detection.

Tune `--max-inference-batch-size` and `--gpu-memory-utilization` for the available GPU.
Shorter `--chunk-seconds` values bound individual requests and repetition, but additional
boundaries can introduce omissions or split words. Measure CER, omissions, and repetition
on representative audio before changing it.

This example was informed by the offline vLLM work in
[qwen3-asr-service](https://github.com/LanceLRQ/qwen3-asr-service) and the user evaluation
in [FunASR #3419](https://github.com/modelscope/FunASR/issues/3419). CER values from
different datasets, transcript alignment rules, or model sizes are not directly comparable.
