# MOSS-Transcribe-Diarize in the FunASR ecosystem

[中文](./moss_transcribe_diarize_zh.md)

This guide connects the third-party
[OpenMOSS/MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)
model to the FunASR deployment ecosystem. The model is published by OpenMOSS
under Apache-2.0; it is not a FunASR model. FunASR provides an adapter for its
public Transformers, vLLM, and SGLang Omni interfaces while retaining the
OpenMOSS model name, license, and upstream revision.

MOSS-Transcribe-Diarize jointly generates transcription, timestamps, and
speaker labels such as `[S01]`. An application therefore does not need to
assemble an external VAD, ASR, and diarization pipeline. This is a deployment
property, not a claim that the model has no internal segmentation or chunking.

## Pinned sources

This guide was verified against:

- source: `OpenMOSS/MOSS-Transcribe-Diarize@cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3`
- model: `OpenMOSS-Team/MOSS-Transcribe-Diarize@e8681d68e7042738ffca8ac8212bc8fcb1131ab8`
- license: Apache-2.0 in the pinned upstream source and model metadata
- vLLM release: `v0.27.1@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`
- vLLM CUDA 12.9 x86_64 wheel SHA-256:
  `bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b`

Pin all three revisions. The model uses `trust_remote_code`; do not execute a
floating model revision in a production service.

## FunASR AutoModel contract

Use an isolated Python 3.10+ environment with Transformers 5.6 or newer for
the local backend. MOSS performs long-form transcription and speaker
diarization in one generation, so do **not** pass `vad_model` or `spk_model`.
External VAD segmentation would break the model's global speaker identity
across chunks.

```python
from funasr import AutoModel

model = AutoModel(
    model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
    model_revision="e8681d68e7042738ffca8ac8212bc8fcb1131ab8",
    backend="hf",
    device="cuda:0",
    dtype="bf16",
    attn_implementation="sdpa",
    disable_update=True,
)

result = model.generate("audio.wav", max_new_tokens=5120)[0]
print(result["text"])
for segment in result["sentence_info"]:
    print(segment["start"], segment["end"], segment["spk"], segment["text"])
```

The adapter preserves the raw tagged generation in `raw_text` and returns the
common FunASR fields:

- `text`: readable transcript without MOSS control tags;
- `timestamp`: segment-level `[start_ms, end_ms]` pairs;
- `sentence_info`: `start`, `end`, `text`, `sentence`, `spk`, and `timestamp`;
- `raw_text`: exact `[start][Sxx]text[end]` generation for auditing.

If the parser cannot prove the tagged structure, it leaves the model text
visible in `text` and `raw_text` and returns empty timestamp/segment arrays
instead of silently inventing speaker metadata.

The same result contract can wrap an already running vLLM service without
downloading local weights:

```python
from funasr import AutoModel

model = AutoModel(
    model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
    backend="vllm",
    vllm_base_url="http://127.0.0.1:8898/v1",
    vllm_model="moss-transcribe-diarize",
    vllm_response_format="diarized_json",
    disable_update=True,
)
result = model.generate("audio.wav", max_completion_tokens=8192)[0]
```

The vLLM adapter sends the documented OpenAI-compatible multipart request and
normalizes the official speaker-attributed `segments` directly into
`sentence_info`. Authentication can be supplied with `vllm_api_key`; keep it
in a secret store rather than source code.

`vllm_response_format="diarized_json"` is the structured production path on
the pinned vLLM revision. The compatibility default remains `json` so existing
clients keep the exact tagged generation in `raw_text`; in structured mode,
`raw_text` is the cleaned `text` returned by vLLM and the authoritative speaker
metadata is in `sentence_info`.

## Choose a serving path

| Path | Environment | Response contract | Use when |
|---|---|---|---|
| vLLM | CUDA 12 (`cu129`) or CUDA 13 (`cu130`) | `response_format=diarized_json` returns OpenAI-compatible speaker segments; `response_format=json` keeps raw `[start][Sxx]text[end]` text | You already operate vLLM or need its scheduler |
| SGLang Omni | CUDA 13 in the current upstream guide | `response_format=verbose_json` returns parsed segments | You need structured speaker segments directly from the API |
| Transformers | PyTorch process | Python objects and raw tagged text | Evaluation, debugging, or custom preprocessing |
| moss-transcribe.cpp / LocalAI | C++17, ggml, and GGUF on CPU or ggml GPU backends | LocalAI's OpenAI-compatible transcription endpoint | You need quantized GGUF or edge deployment without Python/PyTorch on the inference host |

The two HTTP backends share `/v1/audio/transcriptions`, but their documented
response formats are not interchangeable. The pinned vLLM revision supports
`diarized_json`; test the exact server revision before promising `segments`.

## vLLM

Create an isolated environment and install the verified vLLM release:

```bash
uv venv --python 3.12 .venv-moss
curl -fL \
  https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl \
  -o vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
echo "bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b  vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  | sha256sum -c -
uv pip install --python .venv-moss/bin/python --torch-backend=auto \
  "vllm[audio] @ file://$PWD/vllm-0.27.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
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

Submit audio and validate the official speaker-attributed response:

```bash
curl -fsS http://127.0.0.1:8898/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=moss-transcribe-diarize \
  -F response_format=diarized_json \
  -F max_completion_tokens=8192 \
  -F temperature=0 \
  | tee moss-transcription.json

python - <<'PY'
import json

with open("moss-transcription.json", encoding="utf-8") as stream:
    payload = json.load(stream)
text = payload.get("text", "")
segments = payload.get("segments", [])
assert text.strip(), payload
assert segments, payload
assert all(
    isinstance(item.get("speaker"), str)
    and item.get("text")
    and item.get("start") <= item.get("end")
    for item in segments
), payload
print(text, sorted({item["speaker"] for item in segments}))
PY
```

For an audit of the model's compact tagged generation, repeat the request with
`response_format=json` and verify `[S01]` appears in the returned `text`.

For long recordings, test `max_completion_tokens` against the longest expected
meeting. The FunASR vLLM adapter accepts both the OpenAI-compatible
`max_completion_tokens` name and the compatibility alias `max_new_tokens`; the
native name wins when both are present. A larger limit can increase memory use
and tail latency.

### Reproduced vLLM contract

The structured-response validation used vLLM 0.27.1, Torch
`2.13.0+cu129`, and one H100 80GB. The 15.1685-second two-speaker probe
(`43dccc068506439cb633b382b6b98185baa837363d08cc5f7152ca89b0fdc3c8`)
returned two `diarized_json` segments labelled `S01` and `S02`; the same
request through the FunASR adapter returned two monotonic `sentence_info`
entries with the same speakers.

The same environment also processed the two real Chinese long recordings from
FunASR #3539:

- the 309.600-second sample
  (`6561ee553c8f762aac4ebd65439d3414820761b547fa3a2edcea43b86a2abc02`)
  returned 158 segments through 309.41 seconds with the default 5120-token
  limit in 7.459 seconds;
- the 379.664-second sample
  (`779899a3ce937dd7352b4db1ea53e3f6aa2cfef7109de0249082223c936f9372`)
  truncated at 354.98 seconds with the default 5120-token limit, causing an
  HTTP 400 for incomplete `diarized_json` tags. With
  `max_completion_tokens=8192`, it returned 224 segments through 378.55
  seconds in 10.516 seconds.

MOSS corrected several proper-noun and near-homophone errors named by the
reporter and preserved the semantic continuity of the foundation-statement
phrase. It still emitted some short standalone segments, so this does not prove
the subtitle-grouping issue is resolved. These measurements establish an exact
input completeness boundary, not accuracy, throughput, or production capacity.

Earlier raw-response validation on vLLM
`0.23.1rc1.dev949+g68b4a1d58`, Torch `2.11.0+cu129`, and one H100 80GB found:

- the bundled 6.000-second sample
  (`ea03e1f473ad1618a03da3327a545369cb8f6f06cb0f4115535e5a866167d47e`)
  returned HTTP 200 and `[0.96][S01]... [5.94]`;
- an A + 0.8-second silence + B + 0.8-second silence + A probe
  (`dbb32bcfed2e8226bedf64248a9f4a44685b293a4696d18fb4cfa701b04db912`)
  returned HTTP 200 with `S01 -> S02 -> S01` and timestamps through 19.08 seconds.

This proves the pinned API and speaker-return contract on those inputs. It is
not a diarization accuracy result, overlap test, throughput benchmark, or
production capacity claim.

The older nightly commit
`68b4a1d582818e67adc903bf1b8fc5a5447da2fa` predates vLLM
[`#48543`](https://github.com/vllm-project/vllm/pull/48543). It accepts
`response_format=json` but returns HTTP 400 for `diarized_json`; upgrade to the
verified release instead of silently retrying an expensive transcription.

The FunASR adapter was also tested on both `backend="hf"` and
`backend="vllm"` with model revision
`e8681d68e7042738ffca8ac8212bc8fcb1131ab8`. The 15.1685-second two-speaker
probe (`43dccc068506439cb633b382b6b98185baa837363d08cc5f7152ca89b0fdc3c8`)
returned two monotonic segments labelled `S01` and `S02` through the common
`AutoModel` result contract. The temporary vLLM worker was stopped after the
test; this is a contract smoke test, not an accuracy benchmark.

## moss-transcribe.cpp and LocalAI

[`localai-org/moss-transcribe.cpp`](https://github.com/localai-org/moss-transcribe.cpp)
is a third-party C++17/ggml reimplementation maintained by the LocalAI team and
licensed under MIT. The original OpenMOSS model weights remain Apache-2.0. It is
not a FunASR `AutoModel` backend and is not interchangeable with the Python
adapter above.

Choose this path for GGUF quantization, CPU inference, or ggml
CUDA/Metal/Vulkan/HIP backends. LocalAI
`master@a7cc5873ef5b7c909fc9ff7d349d51738ba9bb05` includes a
`moss-transcribe-cpp` backend and Hugging Face importer. The backend pins
`moss-transcribe.cpp@190a569c13b4b247450f2fb3b2a431244e84833e`, while the
importer recognizes `huggingface://mudler/moss-transcribe.cpp-gguf` and selects
Q5_K by default. Before production, follow LocalAI's documentation and verify
the backend package, GGUF SHA-256, actual hardware backend, and
`/v1/audio/transcriptions` response. Do not treat upstream README performance
claims as measurements of your host.

## SGLang Omni

Follow the pinned upstream SGLang Omni installation guide for CUDA 13, then
download the immutable model snapshot and serve the local directory:

```bash
git clone https://github.com/sgl-project/sglang-omni.git
git -C sglang-omni checkout 3f819f9cdae3d4eeec22f73306c9067a1ec2542e
```

This source pin includes the transcription API's `max_new_tokens` forwarding.
The original #914 merge predates that request field, so it is not sufficient
for the long-audio command below even though its published H100 benchmark
remains useful upstream evidence.

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

Verify every segment has start/end timing and non-empty text. In SGLang Omni's
current `verbose_json` contract, the speaker identifier is retained as the
`[Sxx]` prefix in `segments[].text`; it is not a separate `speaker` field.
Parse and validate that prefix before wiring the response into subtitles,
meeting notes, or analytics.

The FunASR adapter performs that validation and maps the official SGLang
segments into the same `sentence_info` contract as the HF and vLLM paths:

```python
from funasr import AutoModel

model = AutoModel(
    model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
    backend="sglang",
    sglang_base_url="http://127.0.0.1:8898/v1",
    sglang_model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
    max_new_tokens=65536,
    disable_update=True,
)
result = model.generate(input="audio.wav", max_new_tokens=65536)[0]
for segment in result["sentence_info"]:
    print(segment["start"], segment["end"], segment["spk"], segment["text"])
```

Do not pass `vad_model` or `spk_model`: MOSS performs segmentation and
anonymous speaker attribution jointly, and external splitting can destroy
speaker consistency across long turns. The adapter preserves the upstream
tagged transcript in `raw_text`, strips only the validated `[Sxx]` prefix from
each normalized segment, and fails closed if SGLang omits that prefix.

The native runtime was merged in SGLang Omni
[#914](https://github.com/sgl-project/sglang-omni/pull/914). Its single-H100
Seed-TTS EN benchmark completed 1088/1088 clips with no request failures. WER
was measured only after removing timestamp and speaker markup from
single-speaker English clips, so it does not evaluate diarization or timestamp
accuracy and is not a production capacity promise.

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
