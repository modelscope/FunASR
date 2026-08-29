# MOSS-Transcribe-Diarize in the FunASR ecosystem

[中文](./moss_transcribe_diarize_zh.md)

This guide connects the third-party
[OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
model to the FunASR deployment ecosystem. The model is published by OpenMOSS
under Apache-2.0; it is not a FunASR model and is not registered in FunASR
`AutoModel`.

MOSS-Transcribe-Diarize jointly generates transcription, timestamps, and
speaker labels such as `[S01]`. An application therefore does not need to
assemble an external VAD, ASR, and diarization pipeline. This is a deployment
property, not a claim that the model has no internal segmentation or chunking.

## Pinned sources

This guide was verified against:

- source: `OpenMOSS/MOSS-Transcribe-Diarize@cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3`
- model: `OpenMOSS-Team/MOSS-Transcribe-Diarize@e8681d68e7042738ffca8ac8212bc8fcb1131ab8`
- license: Apache-2.0 in the pinned upstream source and model metadata
- vLLM nightly index: `68b4a1d582818e67adc903bf1b8fc5a5447da2fa`

Pin all three revisions. The model uses `trust_remote_code`; do not execute a
floating model revision in a production service.

## Choose a serving path

| Path | Environment | Response contract | Use when |
|---|---|---|---|
| vLLM | CUDA 12 (`cu129`) or CUDA 13 (`cu130`) | `response_format=json` returns raw `[start][Sxx]text[end]` text | You already operate vLLM or need its scheduler |
| SGLang Omni | CUDA 13 in the current upstream guide | `response_format=verbose_json` returns parsed segments | You need structured speaker segments directly from the API |
| Transformers | PyTorch process | Python objects and raw tagged text | Evaluation, debugging, or custom preprocessing |

The two HTTP backends share `/v1/audio/transcriptions`, but their documented
response formats are not interchangeable. Do not promise vLLM `segments`
without testing the exact vLLM revision.

## vLLM

Create an isolated environment and install the upstream-pinned nightly build:

```bash
uv venv --python 3.12 .venv-moss
uv pip install --python .venv-moss/bin/python -U 'vllm[audio]' \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129
```

The `audio` extra is required. A plain `vllm` install can start the server but
returns HTTP 400 (`Invalid or unsupported audio file`) because no audio decoder
is installed.

Start the service with the immutable model revision:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-moss/bin/vllm serve \
  OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --revision e8681d68e7042738ffca8ac8212bc8fcb1131ab8 \
  --served-model-name moss-transcribe-diarize \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 8898
```

Submit audio and validate the speaker-tagged text:

```bash
curl -fsS http://127.0.0.1:8898/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=moss-transcribe-diarize \
  -F response_format=json \
  -F temperature=0 \
  | tee moss-transcription.json

python - <<'PY'
import json

with open("moss-transcription.json", encoding="utf-8") as stream:
    payload = json.load(stream)
text = payload.get("text", "")
assert text.strip(), payload
assert "[S01]" in text, text
print(text)
PY
```

For long recordings, test `max_completion_tokens` against the longest expected
meeting. A larger limit can increase memory use and tail latency.

### Reproduced vLLM contract

The FunASR ecosystem validation used vLLM
`0.23.1rc1.dev949+g68b4a1d58`, Torch `2.11.0+cu129`, and one H100 80GB:

- the bundled 6.000-second sample
  (`ea03e1f473ad1618a03da3327a545369cb8f6f06cb0f4115535e5a866167d47e`)
  returned HTTP 200 and `[0.96][S01]... [5.94]`;
- an A + 0.8-second silence + B + 0.8-second silence + A probe
  (`dbb32bcfed2e8226bedf64248a9f4a44685b293a4696d18fb4cfa701b04db912`)
  returned HTTP 200 with `S01 -> S02 -> S01` and timestamps through 19.08 seconds.

This proves the pinned API and speaker-return contract on those inputs. It is
not a diarization accuracy result, overlap test, throughput benchmark, or
production capacity claim.

## SGLang Omni

Follow the pinned upstream SGLang Omni installation guide for CUDA 13, then
download the immutable model snapshot and serve the local directory:

```bash
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --revision e8681d68e7042738ffca8ac8212bc8fcb1131ab8 \
  --local-dir .models/moss-transcribe-diarize

sgl-omni serve \
  --model-path .models/moss-transcribe-diarize \
  --port 8898 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.80
```

Request parsed segments:

```bash
curl -fsS http://127.0.0.1:8898/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F response_format=verbose_json
```

Verify every segment has start/end timing, text, and the expected speaker
field before wiring the response into subtitles, meeting notes, or analytics.

## Production validation

Use real multi-speaker audio, not only a short single-speaker smoke sample:

- measure speaker consistency across long turns and speaker returns;
- test overlap, crosstalk, music, noise, and long silence;
- verify timestamp monotonicity and full-audio coverage;
- record GPU, CUDA, Torch, backend commit, model revision, audio duration,
  generation limit, wall latency, and peak memory;
- keep authentication, TLS, request limits, and retention policy at the API
  gateway; bind the model worker to a private address.

When reporting a problem to FunASR, state that this is the OpenMOSS third-party
path and include the exact backend and upstream revisions. Model architecture or
weight issues belong upstream; deployment-center contract and documentation
issues can be filed in FunASR.
