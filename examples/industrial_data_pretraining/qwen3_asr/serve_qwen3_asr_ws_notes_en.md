# serve_qwen3_asr_ws.py — Notes & Known Issues

This document records the confusing points / gotchas of `serve_qwen3_asr_ws.py`. The code
only keeps one-line comments pointing here; details live in this file.

---

## 1. About VAD

> The open-source repo github.com/QwenLM/Qwen3-ASR contains nothing about `VAD`. However,
> the commercial [Qwen3-ASR docs/examples](https://help.aliyun.com/zh/model-studio/qwen-asr-realtime-interaction-process)
> **do have VAD settings**, e.g.:
> ```json
> "turn_detection": { "type": "server_vad", "threshold": 0.2, "silence_duration_ms": 800 }
> ```

VAD actually serves two completely different purposes in ASR, which are easy to conflate:

- **A. Segmentation VAD (feeding chunks to a non-streaming encoder)**: models like
  Fun-ASR-Nano have a **non-streaming** encoder (it needs to see a whole segment at once),
  so a VAD is required to cut the continuous audio into sentences, each encoded/decoded as a
  whole. This kind of VAD is **technically mandatory** for such models — without cutting,
  there is no way to encode.

- **B. Endpoint / turn-detection VAD (deciding "has this turn finished?")**: detects speaker
  pauses (e.g. 800ms of silence) to decide "a sentence/turn has ended", triggering "lock the
  text / emit is_final / time to respond". This is a **product-behavior** need, unrelated to
  whether the encoder is streaming.

For **Qwen3-ASR's open-source streaming API** (this service uses `qwen-asr[vllm]`'s
`init_streaming_state` / `streaming_transcribe`):

- **No need for type-A (segmentation) VAD**: it is **incremental** streaming — each call only
  consumes the newly added slice of audio, the state rolls forward, transcribing continuously;
  there is no "cut first, then decode" step. "Which characters are final vs. still changing" is
  expressed by `unfixed_chunk_num` / `unfixed_token_num` (the last N chunks/tokens are
  "unfixed" and may be corrected by later audio; the rest are treated as confirmed) — this acts
  as a built-in partial/locking mechanism that replaces type-A VAD's segmentation role. That is
  why you can't find `vad` in the open-source streaming API: it simply **doesn't do
  segmentation at this layer**.

- **Still needs type-B (endpoint) VAD — the open-source API just doesn't ship one**:
  automatically deciding "user paused = this turn ended" is not handled by
  `streaming_transcribe` itself. The commercial service wraps a `server_vad` layer **outside**
  the ASR (the `turn_detection` block above) to do it. This service currently uses an explicit
  **`STOP`** message from the client to substitute for that endpoint decision (in the bench,
  STOP is sent once the audio finishes). **To auto-segment/auto-endpoint in a real scenario,
  you need to add a VAD / endpoint detector outside this service** (the same role as commercial
  `server_vad`), rather than looking for one inside Qwen3-ASR — its open-source streaming API
  does not include this layer.

**In one line**: Qwen3-ASR's incremental streaming **removes the "segmentation VAD" (A)**, but
the **"endpoint / turn VAD" (B) responsibility still exists** — the commercial version
implements it via `server_vad`, while this service substitutes a manual `STOP`. The two are not
contradictory.

### 1.1 Official corroboration: the commercial Qwen-ASR-Realtime "VAD mode / Manual mode"

Aliyun Bailian's real-time ASR (Qwen-ASR-Realtime) docs explicitly split "who does the
segmentation" into two modes — essentially whether `session.turn_detection` is on or off:

- **VAD mode (default, `turn_detection` set to server_vad)**: the server automatically detects
  speech start/end to segment; the client just keeps streaming audio, and the server returns the
  final result automatically when it "detects a sentence has ended". In the flow the server emits
  `input_audio_buffer.speech_started` / `speech_stopped` events — this is exactly the **type-B
  endpoint VAD** above, implemented by that server-side (server_vad) layer, **not** the ASR core
  doing segmentation.

- **Manual mode (`turn_detection` set to null)**: the **client** controls segmentation — after
  sending a full sentence of audio, the client sends `input_audio_buffer.commit` to tell the
  server the boundary. Suitable for scenarios where the client can clearly determine sentence
  boundaries (e.g. "push-to-talk", sending a voice message in a chat app).

> Mapping: this service `serve_qwen3_asr_ws.py` uses an explicit client **`STOP`** to mark the
> end of a turn, which is equivalent to the commercial **Manual mode** (`turn_detection=null`,
> client-controlled boundaries). To make it "server auto-segmentation", you would implement that
> endpoint-detection layer of the commercial **VAD mode** (server_vad) — added outside this
> service's incremental transcription, not found inside the Qwen3-ASR transcription core.
>
> Docs: Real-time Speech Recognition (Qwen-ASR-Realtime) interaction flow
> (help.aliyun.com/zh/model-studio/qwen-asr-realtime-interaction-process)

### 1.2 `chunk-size-sec` controls the streaming chunk size

`chunk-size-sec` controls the streaming chunk size (default 2.0). Smaller = faster/more frequent
output, but higher concurrency cost (measured on L20, 29s audio, 48-way concurrency: 1.0 →
all failed, 2.0 → all passed).

---

## 2. Use vllm 0.14, not 0.19 (rope_scaling / thinker_config warnings)

This service needs vllm acceleration via `qwen-asr[vllm]`, which
[pins `vllm==0.14.0`](https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml). Switching to
a newer vllm (e.g. 0.19.x) will produce at startup:

```
Unrecognized keys in `rope_scaling` for 'rope_type'='default':
    {'mrope_section', 'mrope_interleaved', 'interleaved'}
thinker_config is None. Initializing thinker model with default values
```

**Root cause**: between vllm 0.14 → 0.19 (transformers is 4.57.6 in both, so it's ruled out),
the config parsing in `patch_rope_scaling_dict` rewrites `rope_type` from `'mrope'` to
`'default'` (treating mrope as legacy and assuming vllm internally consumes the `mrope_section`
etc. fields):

```python
elif rope_scaling["rope_type"] == "mrope":
    assert "mrope_section" in rope_scaling
    rope_scaling["rope_type"] = "default"   # ← rewrite
```

But Qwen3-ASR's own `Qwen3ASRThinkerTextRotaryEmbedding` expects to read `"mrope"` from
`rope_scaling` to take the multimodal RoPE branch:

```python
self.rope_type = config.rope_scaling.get("rope_type", "default")
```

After vllm rewrites it to `"default"`, it takes the plain RoPE branch, and the keys
`mrope_section` / `mrope_interleaved` / `interleaved` are left unclaimed → the "Unrecognized
keys" warning, with degraded multimodal (audio/text) positional encoding. The
`thinker_config is None` line has the same origin: 0.19's load path doesn't correctly parse
Qwen3-ASR's thinker sub-config and falls back to default parameters.

**Impact & decision**: on 0.19 the service "starts and does emit text" (both lines are
WARNING/INFO, not ERROR), and spot-checked transcripts "look fine"; but the degraded positional
encoding may be harmful for long/complex audio, and **no CER quantitative comparison was done**,
so equivalence cannot be established. To be safe, stick with the `vllm==0.14.0` that
`qwen-asr[vllm]` ships.

### 2.1 vllm acceleration requires Qwen3ASRModel

funasr's `AutoModelVLLM` cannot accelerate Qwen3-ASR; you must use
`from qwen_asr import Qwen3ASRModel`.

> This answers the question in `#3026`.

### 2.2 about download model weights

As in [Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR#released-models-description-and-download) suggests: when the runtime can't download online, **pre-download the weights to a local dir, then pass that local path as `--model`**:

 ```bash 
# Option 1: ModelScope (recommended in Mainland China) 
pip install -U modelscope 
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./Qwen3-ASR-1.7B 
# Option 2: Hugging Face 
pip install -U "huggingface_hub[cli]" 
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B 

#Then start the ws server with the pre-downloaded weights 
python serve_qwen3_asr_ws.py --model ./Qwen3-ASR-1.7B ...
 ```

Note: **On `VLLM_USE_MODELSCOPE=True` With this var set + `pip install modelscope`, **only take over part of the download — the vLLM layer fetches config / tokenizer / merges / vocab / `model.safetensors.index.json` from ModelScope (log: `Downloading Model from https://www.modelscope.cn ... Finish downloading 10 files`), but the step that fetches the actual weights `model.safetensors` **still goes back to huggingface.co**:

**This should be a Qwen-ASR bug**:

```
Downloading Model from https://www.modelscope.cn to directory: /home/vllm/.cache/modelscope/hub/models/Qwen/Qwen3-ASR-1.7B
2026-06-28 10:15:56,301 - modelscope - INFO - Got 10 files, start to download ...
Downloading [configuration.json]: 100%|
...
INFO 06-28 10:15:59 [model.py:530] Resolved architecture: Qwen3ASRForConditionalGeneration
'(MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /Qwen/Qwen3-ASR-1.7B/resolve/main/model.safetensors (Caused by NewConnectionError("HTTPSConnection(host=\'huggingface.co\', port=443): Failed to establish a new connection: [Errno 101] Network is unreachable"))'), '(Request ID: f4f58ad5-e161-42ec-baad-cb2b8dbb63b9)')' thrown while requesting HEAD https://huggingface.co/Qwen/Qwen3-ASR-1.7B/resolve/main/model.safetensors
```

---

## 3. tokenizer `fix_mistral_regex` warning (harmless, ignorable)

At startup you may see:

```
The tokenizer you are loading from '.../Qwen3-ASR-1.7B' with an incorrect regex
pattern ... This will lead to incorrect tokenization. You should set the
`fix_mistral_regex=True` flag when loading this tokenizer to fix this issue.
```

**Cause**: Qwen3-ASR's tokenizer reuses a tokenizer implementation with a known regex issue; the
underlying library detects that regex pattern and warns, suggesting `fix_mistral_regex=True` to
fix the splitting. This service loads the model via qwen-asr's high-level API and does not
construct the tokenizer directly, so it does not expose this flag — hence the warning prints
as-is.

**Impact**: measured to have no visible effect on Chinese ASR transcripts (**spot-checked**
multiple transcripts as normal; no CER quantification). That regex fix mainly affects the
boundary splitting of certain special tokens, and no difference was observed on the speech
transcription path. It's "notice-level" noise and can be ignored. To eliminate it entirely you'd
pass `fix_mistral_regex=True` when loading the tokenizer at a lower level, but the qwen-asr
high-level API doesn't directly support that, and there's no measured need.

---

### Aside: two other startup log lines (both harmless)

- `Error retrieving safetensors: Repo id must be in the form ...`: it treats the local model
  path as an HF repo id to fetch online metadata, fails, retries twice, and falls back to local
  loading — no functional impact. Can be silenced with the env var `HF_HUB_OFFLINE=1`.

- `Downcasting torch.float32 to torch.bfloat16`: weights are stored as fp32 and loaded as bf16,
  normally saving VRAM / speeding things up; bf16 has the same exponent width as fp32, so
  precision loss is negligible. This is INFO, not an error.
