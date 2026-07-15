#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "$ROOT/../.." && pwd)
SCRIPT="$ROOT/download-funasr-model.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/bin"
cat >"$TMP/bin/hf" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >>"$HF_LOG"

if [[ "${HF_SKIP_CREATE:-0}" == 1 ]]; then
  exit 0
fi

out=
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--local-dir" ]]; then
    out="${args[$((i + 1))]}"
    break
  fi
done

[[ -n "$out" ]]
mkdir -p "$out"
for arg in "${args[@]}"; do
  case "$arg" in
    '*.gguf') : >"$out/all.gguf" ;;
    *.gguf) : >"$out/$arg" ;;
  esac
done
EOF
chmod +x "$TMP/bin/hf"

export PATH="$TMP/bin:$PATH"
export HF_LOG="$TMP/hf.log"

reset_case() {
  : >"$HF_LOG"
  rm -rf "$TMP/out"
}

assert_log() {
  grep -F -- "$1" "$HF_LOG" >/dev/null
}

assert_no_log() {
  if grep -F -- "$1" "$HF_LOG" >/dev/null; then
    printf 'unexpected log entry: %s\n' "$1" >&2
    cat "$HF_LOG" >&2
    exit 1
  fi
}

reset_case
bash "$SCRIPT" sensevoice "$TMP/out" >/dev/null
assert_log "download FunAudioLLM/SenseVoiceSmall-GGUF sensevoice-small-q8.gguf --local-dir $TMP/out"
assert_log "download FunAudioLLM/fsmn-vad-GGUF fsmn-vad.gguf --local-dir $TMP/out"
assert_no_log "--include *.gguf"

reset_case
bash "$SCRIPT" paraformer "$TMP/out" f16 >/dev/null
assert_log "download FunAudioLLM/Paraformer-GGUF paraformer-f16.gguf --local-dir $TMP/out"

reset_case
bash "$SCRIPT" nano "$TMP/out" >/dev/null
assert_log "download FunAudioLLM/Fun-ASR-Nano-GGUF funasr-encoder-f16.gguf qwen3-0.6b-q8_0.gguf --local-dir $TMP/out"

reset_case
bash "$SCRIPT" nano "$TMP/out" q4km >/dev/null
assert_log "download FunAudioLLM/Fun-ASR-Nano-GGUF funasr-encoder-f16.gguf qwen3-0.6b-q4km.gguf --local-dir $TMP/out"

reset_case
bash "$SCRIPT" sensevoice "$TMP/out" all >/dev/null
assert_log "download FunAudioLLM/SenseVoiceSmall-GGUF --include *.gguf --local-dir $TMP/out"

reset_case
bash "$SCRIPT" fsmn-vad "$TMP/out" >/dev/null
assert_log "download FunAudioLLM/fsmn-vad-GGUF fsmn-vad.gguf --local-dir $TMP/out"
[[ $(wc -l <"$HF_LOG") -eq 1 ]]

reset_case
if bash "$SCRIPT" sensevoice "$TMP/out" q4km >/dev/null 2>&1; then
  echo "sensevoice accepted an unsupported q4km variant" >&2
  exit 1
fi
[[ ! -s "$HF_LOG" ]]

mkdir -p "$TMP/empty-bin"
if PATH="$TMP/empty-bin" /bin/bash "$SCRIPT" sensevoice "$TMP/out" >"$TMP/stdout" 2>"$TMP/stderr"; then
  echo "download succeeded without a Hugging Face CLI" >&2
  exit 1
fi
[[ ! -s "$TMP/stdout" ]]
grep -F "need the Hugging Face CLI" "$TMP/stderr" >/dev/null

reset_case
if HF_SKIP_CREATE=1 bash "$SCRIPT" sensevoice "$TMP/out" >"$TMP/stdout" 2>"$TMP/stderr"; then
  echo "download succeeded without producing a GGUF file" >&2
  exit 1
fi
grep -F "no GGUF files found in $TMP/out" "$TMP/stderr" >/dev/null

assert_readme_quickstart() {
  local readme=$1
  if ! grep -F "bash download-funasr-model.sh sensevoice ./gguf" "$readme" >/dev/null; then
    printf 'missing default SenseVoice download command in %s\n' "$readme" >&2
    exit 1
  fi
  if ! grep -F "llama-funasr-sensevoice -m ./gguf/sensevoice-small-q8.gguf --vad ./gguf/fsmn-vad.gguf -a audio.wav" "$readme" >/dev/null; then
    printf 'README command does not run the default sensevoice-small-q8.gguf in %s\n' "$readme" >&2
    exit 1
  fi
  if grep -F "SenseVoiceSmall-f16.gguf" "$readme" >/dev/null; then
    printf 'stale SenseVoice filename in %s\n' "$readme" >&2
    exit 1
  fi
}

assert_readme_quickstart "$REPO_ROOT/README.md"
assert_readme_quickstart "$REPO_ROOT/README_zh.md"

echo "download-funasr-model contract tests passed"
