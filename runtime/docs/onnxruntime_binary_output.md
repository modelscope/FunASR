# ONNX Runtime binary JSONL output

The native ONNX Runtime binaries log results by default. Add `--output-format jsonl` when another process needs one machine-readable record for every completed input.

## Offline recognition

From `runtime/onnxruntime`, run:

```shell
./build/bin/funasr-onnx-offline \
  --model-dir /path/to/timestamp-capable-model \
  --wav-path /path/to/audio.wav \
  --output-format jsonl \
  > results.jsonl 2> runtime.log
```

The same option is available on `funasr-onnx-offline-rtf`.

## Two-pass recognition

```shell
./build/bin/funasr-onnx-2pass \
  --model-dir /path/to/offline-model \
  --online-model-dir /path/to/online-model \
  --vad-dir /path/to/vad-model \
  --punc-dir /path/to/online-punctuation-model \
  --wav-path /path/to/audio.wav \
  --mode 2pass \
  --output-format jsonl \
  > results.jsonl 2> runtime.log
```

The same option is available on `funasr-onnx-2pass-rtf`. Valid modes are `offline`, `online`, and `2pass`. The native two-pass runtime requires both a VAD model and an online punctuation model for successful inference.

## Record schema

Each stdout line is one JSON object:

```json
{"key":"utt-001","mode":"offline","stamp_sents":[{"end":920,"start":0,"text_seg":"hello world"}],"text":"hello world","timestamp":[[0,420],[460,920]]}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `key` | string | The first column of a `wav.scp` row, or `wav_default_id` for a direct file input. |
| `mode` | string | `offline`, `online`, or `2pass`. |
| `text` | string | The completed transcript. In `2pass` mode this is the corrected offline transcript. |
| `timestamp` | array | The native token/word timestamp array returned by the model. |
| `stamp_sents` | array | Native sentence timestamp objects when the binary and model provide them. Two-pass binaries currently return an empty array here. |

Use a timestamp-capable model to receive timestamp values. If a model returns no timestamp, or the native payload is empty, malformed, or not a JSON array, the corresponding field is `[]`.

## Streaming and batch behavior

- JSON records are written only to stdout. Runtime diagnostics and existing progress logs remain on stderr.
- A `wav.scp` input produces one line per successfully completed key.
- RTF binaries protect each complete line from interleaving, but worker completion order is not guaranteed. Join records by `key` instead of relying on line order.
- Omitting `--output-format` is equivalent to `--output-format log` and preserves the existing logging behavior.
