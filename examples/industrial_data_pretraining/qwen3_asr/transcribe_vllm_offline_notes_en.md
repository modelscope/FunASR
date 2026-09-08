# Offline long-form Qwen3-ASR with vLLM

`transcribe_vllm_offline.py` defaults to Qwen3-ASR's native `Qwen3ASRModel.LLM` backend. It does
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

## MOSS Offline Transcription and Anonymous Speakers

Select `--engine moss` for within-recording speaker attribution. This is a separate
HTTP path for the third-party MOSS-Transcribe-Diarize model, not speaker labels attached
to Qwen3-ASR text. It reuses the FunASR `AutoModel` MOSS adapter and uploads the whole
recording once: no Qwen3 180-second splitter, external VAD, second speaker model, or
client-side MOSS weight download.

Start a service in a separate GPU environment using the
[MOSS vLLM deployment guide](../../../docs/moss_transcribe_diarize.md#vllm), and verify
that it returns `diarized_json`. That guide pins server vLLM **0.27.1**. Do not install
it into the Qwen3 `vllm==0.14.0` environment above. The HTTP client itself needs no GPU.

From a FunASR checkout containing this example update, create a separate client environment.
This recipe targets a Linux/Python 3.12 CPU client. FunASR's installation dependencies
do not install Torch automatically, so install it explicitly:

```bash
python3.12 -m venv .venv-moss-client
source .venv-moss-client/bin/activate
python -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .

python examples/industrial_data_pretraining/qwen3_asr/transcribe_vllm_offline.py \
  recording.mp3 \
  --engine moss \
  --vllm-base-url http://127.0.0.1:8898/v1 \
  --served-model moss-transcribe-diarize \
  --request-timeout 600 \
  --max-completion-tokens 8192
```

Use your actual reachable service URL and match `--served-model` to its registered
model name. For authenticated services, set `MOSS_VLLM_API_KEY` through your secret
management mechanism before running; do not place real keys in source or command-line
arguments. Do not pass Qwen3-only `--model`, `--language`, chunking, or GPU options in
this mode. The example is supplied in current source; older PyPI installations may
not contain this mode.

The default output is `recording.moss-vllm.json`; `--output` selects another result file:

- `text`: the MOSS service transcript, never merged with Qwen3 output.
- `sentence_info`: segment `start`, `end`, `text`, and `spk`, with times in milliseconds.
- `timestamp`: the corresponding `[start_ms, end_ms]` pairs.
- `raw_text`: in this structured mode, the cleaned service text, not the raw tagged generation.

The terminal also prints segment times and anonymous labels, such as `10..90 ms [S02] ...`.
HTTP errors or malformed segments raise an error without creating a new success result
or replacing an existing result file. `S01/S02` identify anonymous speakers within one
recording only, not known people, voice verification, or cross-file identities. This is
not a realtime WebSocket path.

`8192` is an adjustable completion limit, not a guarantee of complete transcription for
480-second or longer recordings. Check server context/output limits, the final segment
time, audible tail, and actual speaker attribution. Labels from independently split
files cannot simply be joined; this sample does not invent cross-chunk continuity.
Local HTTP contract tests do not establish model CER, diarization accuracy, or capacity.
