# FunASR Binary JSON Timestamp Output Design

## Context

FunASR issue #3457 asks for text and timestamps from the native binaries so non-Python applications can consume recognition results. The four relevant ONNX Runtime binaries already call `FunASRGetStamp`, and the offline variants also call `FunASRGetStampSents`, but they only write human-oriented glog messages. That output is hard to parse reliably because it includes timestamps, thread ids, and unrelated runtime logs.

## Decision

Add an opt-in `--output-format jsonl` mode to:

- `funasr-onnx-offline`
- `funasr-onnx-offline-rtf`
- `funasr-onnx-2pass`
- `funasr-onnx-2pass-rtf`

The default remains `--output-format log`, preserving every existing invocation and log-oriented workflow. JSON records are written to stdout; glog continues to write diagnostics to stderr.

## Record Contract

Each completed input produces exactly one JSON object followed by one newline:

```json
{"key":"utt-1","mode":"offline","text":"recognized text","timestamp":[[0,320],[320,660]],"stamp_sents":[]}
```

The fields are always present:

- `key`: the `wav.scp` key, or `wav_default_id` for a direct audio path.
- `mode`: `offline`, `online`, or `2pass`.
- `text`: the final transcript. Two-pass mode emits the corrected offline transcript.
- `timestamp`: a JSON array parsed from `FunASRGetStamp`; it is `[]` when the model/runtime returns no valid timestamp array.
- `stamp_sents`: a JSON array parsed from `FunASRGetStampSents`; it is `[]` when unavailable. The current two-pass binary does not expose sentence timestamps, so its field remains an empty array rather than changing type or disappearing.

Malformed native timestamp payloads must not produce malformed JSON or crash result emission; they degrade to an empty array. Text and keys are escaped by nlohmann/json rather than handwritten string manipulation.

## Concurrency

RTF binaries may finish inputs out of order. They must serialize the complete JSON string before taking an output mutex, then write the line while holding that mutex. This guarantees valid, non-interleaved JSONL records without imposing ordered buffering or reducing inference concurrency.

## Components

`runtime/onnxruntime/bin/result-json.{h,cpp}` owns output-format parsing and JSON serialization. The four binary entry points only gather the existing result fields and call that module. `runtime/onnxruntime/bin/tests/result-json-test.cpp` tests real C++ behavior without loading a model. A Python contract test verifies that all four entry points expose the option and that the bilingual documentation names the stable fields.

## Error Handling

Only `log` and `jsonl` are accepted. An unsupported value logs a clear error and exits with status 2 before model initialization. Empty, non-array, or malformed timestamp payloads become `[]`; inference failures retain the existing error path and do not emit a false successful record.

## Documentation

Add a bilingual native-binary output guide with exact commands, stdout/stderr separation, schema definitions, timestamp-capable model requirements, direct-file versus `wav.scp` keys, and the unordered nature of RTF output. Link it from the ONNX Runtime build guide and the English and Chinese runtime indexes.

## Verification

1. Red-first C++ tests cover Unicode/escaping, parsed arrays, empty arrays, malformed payloads, and output-format validation.
2. Red-first Python contract tests cover all four CLI entry points; the repository markdown-link validator covers all three documentation links.
3. Build and run the isolated `funasr-result-json-test` target.
4. Compile all four changed binary translation units through the repository CMake target when the local ONNX Runtime SDK permits it; otherwise record the exact environmental gate and still run syntax checks against the repository headers.
5. Run focused tests, markdown-link validation, formatting, and `git diff --check` before push.
