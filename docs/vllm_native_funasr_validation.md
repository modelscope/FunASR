# Native FunASR on vLLM 0.27.1 validation

Verified on 2026-08-13. This record is a reproducible compatibility and
concurrency probe, not an accuracy benchmark or production capacity claim.

## Pinned stack

- GPU: NVIDIA H100 80GB HBM3; driver 550.127.08
- vLLM: 0.27.1+cu129, official x86_64 release wheel
- vLLM wheel SHA-256: `bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b`
- Torch: 2.13.0+cu129; CUDA available
- Model: `allendou/Fun-ASR-Nano-2512-vllm`
- Model revision: `e718b36e2578203ec893e9b488239225f8d668e2`
- Model weight SHA-256: `96dfbec48282dd24d3334369a01e9e909f321ee39a1b0003c528c5379f68c1a6`
- Audio extra: av 18.1.0, scipy 1.18.0, soundfile 0.14.0, soxr 1.1.0

The checkpoint above is a community conversion. vLLM's FunASR architecture,
transcription endpoint, hotword support, and initialization fix are upstream,
but the converted checkpoint is not an official FunAudioLLM weight release.
Use the [official split-engine guide](vllm_guide.md) when an official-weight
chain is required.

## Server

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/vllm serve \
  allendou/Fun-ASR-Nano-2512-vllm \
  --revision e718b36e2578203ec893e9b488239225f8d668e2 \
  --served-model-name fun-asr-nano \
  --host 127.0.0.1 --port 8899 \
  --dtype float32 \
  --gpu-memory-utilization 0.40 \
  --enforce-eager
```

The engine resolved `FunASRForConditionalGeneration`, retained the 40,960
maximum model length, allocated 17.52 GiB of KV cache (82,016 tokens), and
reported 2.00x maximum concurrency at that length. The cached second startup
spent 20.30 seconds in engine profile, KV-cache creation, and warmup. API
requests were sent only after `/health` returned HTTP 200.

`--gpu-memory-utilization 0.20` is insufficient for the full model length: it
left 1.70 GiB for KV cache while one 40,960-token request required 8.75 GiB.
Do not copy the 0.40 setting blindly; size it against the target GPU and load.

## Inputs and results

All timing is client wall time over localhost after the server was healthy.
Audio loading, model execution, and decoding are included. Model download and
server startup are excluded.

| Probe | Input | Result |
| --- | --- | --- |
| Chinese baseline | 6 s `example/zh.mp3`, SHA-256 `0e64de19e4ff9a02e682955c9112f32d2317cfdbb5bc2f3504664044c993f195` | HTTP 200 in 0.968 s: `开饭时间早上九点至下午五点。` |
| Chinese hotword | Same input; `hotwords=开放时间,开放时间,开放时间` | HTTP 200 in 0.214 s: `开放时间早上九点至下午五点。` |
| Two concurrent requests | 8 s English plus 8 s Japanese examples | Both HTTP 200; 1.123 s total wall time (English 1.036 s, Japanese 1.111 s) |

The English result was `The tribal chieftain called for the boy, and presented
him with fifty pieces of gold.` The Japanese result was
`うちの中学は弁当制で、持っていけない場合は、五十円の学校販売のパンを買う。`

A single `开放时间` hotword did not alter the baseline output. Repeating it
three times changed the ambiguous phrase. This proves the request parameter
reached the generation prompt, but also shows that hotword strength is a policy
to validate on representative audio; it is not a deterministic correction.

## Operational boundaries

- Install `vllm[audio]`; without the audio extra, valid MP3 uploads return HTTP
  400 with `Invalid or unsupported audio file`.
- Pass `language` explicitly for non-English audio. The current vLLM FunASR
  adapter defaults an omitted language to English.
- Keep workers private and put authentication, TLS, rate limits, and audio
  size/duration limits at the gateway.
- Re-run accuracy, latency, concurrency, memory, and hotword tests on the exact
  production GPU, driver, languages, and traffic distribution.
