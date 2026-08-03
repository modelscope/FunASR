# FunASR Binary JSON Timestamp Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give native FunASR ONNX Runtime binaries an opt-in, machine-readable JSONL transcript and timestamp output while preserving default behavior.

**Architecture:** A small shared C++ formatter parses native timestamp strings with nlohmann/json and emits a fixed record schema. Four existing binary entry points add one validated CLI option and call the formatter; RTF callers guard complete-line writes with a mutex. Focused C++ and Python contract tests cover behavior and integration without requiring model downloads.

**Tech Stack:** C++14, nlohmann/json 3.11.2 already fetched by the runtime build, TCLAP, CMake, Python subprocess tests.

## Global Constraints

- `--output-format log` remains the default and preserves current result logging.
- `--output-format jsonl` writes one completed-input record to stdout; diagnostics remain on stderr.
- Every JSON record contains `key`, `mode`, `text`, `timestamp`, and `stamp_sents`.
- Missing, malformed, or non-array native timestamp payloads serialize as `[]`.
- RTF records are atomic but may be emitted out of input order.
- No model download is required for the formatter test.

---

### Task 1: Shared JSON result formatter

**Files:**
- Create: `runtime/onnxruntime/bin/result-json.h`
- Create: `runtime/onnxruntime/bin/result-json.cpp`
- Create: `runtime/onnxruntime/bin/tests/result-json-test.cpp`
- Modify: `runtime/onnxruntime/CMakeLists.txt`
- Modify: `runtime/onnxruntime/bin/CMakeLists.txt`

**Interfaces:**
- Produces: `funasr::OutputFormat ParseOutputFormat(const std::string&)`.
- Produces: `std::string FormatResultJson(const std::string& key, const std::string& mode, const std::string& text, const std::string& timestamp, const std::string& stamp_sents)`.

- [ ] **Step 1: Write the failing C++ test**

Cover `log`/`jsonl`, invalid values, Unicode and quote escaping, valid timestamp arrays, and malformed/empty arrays.

- [ ] **Step 2: Configure and build the test to verify RED**

Run:

```bash
cmake -S runtime/onnxruntime -B /tmp/funasr-3457-build -DFUNASR_BUILD_TESTS=ON -DONNXRUNTIME_DIR=/path/to/onnxruntime
cmake --build /tmp/funasr-3457-build --target funasr-result-json-test -j2
```

Expected: configuration or compilation fails because the formatter and target do not exist.

- [ ] **Step 3: Implement the minimal formatter and test target**

Use `nlohmann::json::parse(raw, nullptr, false)` and accept only arrays; always construct the outer object with nlohmann/json. Add `FUNASR_BUILD_TESTS` defaulting to `OFF`, and register `funasr-result-json-test` only when enabled.

- [ ] **Step 4: Run the C++ test to verify GREEN**

Run:

```bash
cmake --build /tmp/funasr-3457-build --target funasr-result-json-test -j2
/tmp/funasr-3457-build/bin/funasr-result-json-test
```

Expected: exit 0 with all formatter assertions satisfied.

### Task 2: Wire JSONL into four native binaries

**Files:**
- Modify: `runtime/onnxruntime/bin/funasr-onnx-offline.cpp`
- Modify: `runtime/onnxruntime/bin/funasr-onnx-offline-rtf.cpp`
- Modify: `runtime/onnxruntime/bin/funasr-onnx-2pass.cpp`
- Modify: `runtime/onnxruntime/bin/funasr-onnx-2pass-rtf.cpp`
- Create: `runtime/onnxruntime/bin/tests/cli-output-format-test.py`
- Modify: `runtime/onnxruntime/bin/CMakeLists.txt`

**Interfaces:**
- Consumes: `ParseOutputFormat` and `FormatResultJson` from Task 1.
- Produces: `--output-format log|jsonl` on all four binaries.

- [ ] **Step 1: Write the failing executable integration test**

Run each built binary with `--help` and assert that it exposes `--output-format`. Run each binary with its required model/audio arguments pointed at nonexistent paths plus `--output-format yaml`, and assert exit status 2 plus the validation error. This proves the option is wired and rejected before model initialization without scanning implementation text.

- [ ] **Step 2: Run the focused test to verify RED**

Run: `python runtime/onnxruntime/bin/tests/cli-output-format-test.py --bin-dir /tmp/funasr-3457-build/bin`

Expected: failures because all four executables omit the option.

- [ ] **Step 3: Add the option and result emission**

Keep the existing log branch unchanged. In JSONL mode, emit one final record per successful input. Use `offline` for offline binaries and the selected `online`/`offline`/`2pass` value for two-pass binaries. In two-pass mode, use the corrected offline transcript as `text`.

- [ ] **Step 4: Run focused Python and C++ tests**

Run:

```bash
/tmp/funasr-3457-build/bin/funasr-result-json-test
python runtime/onnxruntime/bin/tests/cli-output-format-test.py --bin-dir /tmp/funasr-3457-build/bin
```

Expected: all pass.

### Task 3: Bilingual usage documentation

**Files:**
- Create: `runtime/docs/onnxruntime_binary_output.md`
- Create: `runtime/docs/onnxruntime_binary_output_zh.md`
- Modify: `runtime/onnxruntime/readme.md`
- Modify: `runtime/readme.md`
- Modify: `runtime/readme_cn.md`

**Interfaces:**
- Produces: discoverable English and Chinese guides for the stable record schema.

- [ ] **Step 1: Add the guides and links from the ONNX Runtime build guide and both runtime indexes**

Provide commands for offline and two-pass binaries, a valid one-line JSON example, and explicit empty-array behavior.

- [ ] **Step 2: Run the repository markdown-link validator**

Run:

```bash
python -m pytest -q tests/test_markdown_relative_links.py
```

Expected: all pass.

### Task 4: Full verification, backup, and publication

**Files:**
- Verify all files changed by Tasks 1-3.

**Interfaces:**
- Produces: signed commit, exact-head backup bundle/patch, and a ready PR linked to #3457.

- [ ] **Step 1: Build all changed runtime targets**

Run:

```bash
cmake --build /tmp/funasr-3457-build --target funasr-onnx-offline funasr-onnx-offline-rtf funasr-onnx-2pass funasr-onnx-2pass-rtf -j2
```

Expected: all four binaries link successfully.

- [ ] **Step 2: Run focused and repository quality gates**

Run:

```bash
python runtime/onnxruntime/bin/tests/cli-output-format-test.py --bin-dir /tmp/funasr-3457-build/bin
python -m pytest -q tests/test_markdown_relative_links.py
git diff --check
```

Expected: all tests pass and diff check is clean.

- [ ] **Step 3: Preserve rollback evidence**

Create an exact-head bundle, patch, changed-file copies, build/test logs, and SHA-256 manifest under `/cpfs_speech/user/zhifu.gzf/.cache/funasr-ops/backups/funasr-issue3457-json-timestamps-20260803`.

- [ ] **Step 4: Commit, push, and open the PR**

Create an SSH-signed+DCO commit, push `codex/funasr-binary-json-timestamps-3457-20260803`, open a ready PR that fixes #3457, and monitor all checks/reviews before merging.
