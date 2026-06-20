#!/usr/bin/env bash
# Numerical regression test for the FunASR llama.cpp runtime.
# Runs each available tool on a fixed clip (sample.wav) and diffs the output against
# the frozen golden in golden/. Catches regressions in the ggml graphs, the FSMN-VAD
# state machine, CIF predictor and CTC decode. Golden captured on Linux x86-64 (the
# reference platform) with the f16 GGUFs published on Hugging Face.
#
#   ./run_regression.sh                 # test tools whose binary+model are present (VAD model auto-fetched)
#   RUN_FULL=1 ./run_regression.sh      # also download the ASR GGUFs and test every tool
#   BIN_DIR=/path/to/bin MODELS_DIR=/path/to/gguf ./run_regression.sh
set -u
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RT=$(cd "$DIR/.." && pwd)
BIN="${BIN_DIR:-$RT/build/bin}"
MODELS="${MODELS_DIR:-$DIR/models}"
SAMPLE="$DIR/sample.wav"
RUN_FULL="${RUN_FULL:-0}"
mkdir -p "$MODELS"
pass=0; fail=0; skip=0

bin(){ if [ -x "$BIN/$1" ]; then echo "$BIN/$1"; elif command -v "$1" >/dev/null 2>&1; then echo "$1"; fi; }
# ensure the listed model files exist in $MODELS (download via the repo script if allowed)
ensure_models(){ local key="$1"; shift
  local missing=0; for f in "$@"; do [ -f "$MODELS/$f" ] || missing=1; done
  [ "$missing" = 0 ] && return 0
  { [ "$key" = fsmn-vad ] || [ "$RUN_FULL" = 1 ]; } || return 1   # only auto-fetch the tiny VAD unless RUN_FULL
  "$RT/download-funasr-model.sh" "$key" "$MODELS" >/dev/null 2>&1 || return 1
  for f in "$@"; do [ -f "$MODELS/$f" ] || return 1; done; return 0
}
check(){ local name="$1" golden="$2" got="$3"
  if [ "$got" = "$(cat "$golden")" ]; then echo "  PASS  $name"; pass=$((pass+1))
  else echo "  FAIL  $name"; echo "    expected: $(head -c 80 "$golden")"; echo "    got:      $(printf %s "$got" | head -c 80)"; fail=$((fail+1)); fi
}
skipper(){ echo "  SKIP  $1 ($2)"; skip=$((skip+1)); }

run_tool(){ # name binary golden key  models... -- run...
  local name="$1" b key gold; b=$(bin "$2"); gold="$DIR/golden/$3"; key="$4"; shift 4
  local models=(); while [ "$1" != "--" ]; do models+=("$1"); shift; done; shift
  [ -n "$b" ] || { skipper "$name" "no binary"; return; }
  [ -f "$gold" ] || { skipper "$name" "no golden"; return; }
  ensure_models "$key" "${models[@]}" || { skipper "$name" "model missing (set RUN_FULL=1)"; return; }
  check "$name" "$gold" "$("$@" 2>/dev/null)"
}

echo "== FunASR llama.cpp regression (sample.wav) =="
B=$(bin llama-funasr-vad)        && run_tool vad        llama-funasr-vad        vad.txt        fsmn-vad   fsmn-vad.gguf -- "$B" -m "$MODELS/fsmn-vad.gguf" -a "$SAMPLE"
B=$(bin llama-funasr-sensevoice) && run_tool sensevoice llama-funasr-sensevoice sensevoice.txt sensevoice sensevoice-small-f16.gguf -- "$B" -m "$MODELS/sensevoice-small-f16.gguf" -a "$SAMPLE"
B=$(bin llama-funasr-paraformer) && run_tool paraformer llama-funasr-paraformer paraformer.txt paraformer paraformer-f16.gguf -- "$B" -m "$MODELS/paraformer-f16.gguf" -a "$SAMPLE"
B=$(bin llama-funasr-cli)        && run_tool nano       llama-funasr-cli        nano.txt       nano       funasr-encoder-f16.gguf qwen3-0.6b-q8_0.gguf -- "$B" --enc "$MODELS/funasr-encoder-f16.gguf" -m "$MODELS/qwen3-0.6b-q8_0.gguf" -a "$SAMPLE"

echo "== $pass passed, $fail failed, $skip skipped =="
[ "$fail" = 0 ]
