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
DEFAULT_SENSEVOICE_MODEL=$(awk '$1 == "download" && $2 == "FunAudioLLM/SenseVoiceSmall-GGUF" { print $3; exit }' "$HF_LOG")
[[ -n "$DEFAULT_SENSEVOICE_MODEL" ]]

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
  local section_prefix=$2
  local section
  local quickstart
  local powershell

  if ! section=$(awk -v prefix="$section_prefix" '
    !found && index($0, prefix) == 1 { found = 1 }
    found && /^---$/ { exit }
    found { print }
    END { if (!found) exit 1 }
  ' "$readme"); then
    printf 'missing CPU/edge section in %s\n' "$readme" >&2
    exit 1
  fi

  if ! quickstart=$(printf '%s\n' "$section" | awk '
    !seen && /^```bash$/ { seen = 1; in_block = 1; next }
    in_block && /^```$/ { complete = 1; exit }
    in_block { print }
    END { if (!seen || !complete) exit 1 }
  '); then
    printf 'missing Bash quickstart block in %s\n' "$readme" >&2
    exit 1
  fi

  if ! powershell=$(printf '%s\n' "$section" | awk '
    !seen && /^```powershell$/ { seen = 1; in_block = 1; next }
    in_block && /^```$/ { complete = 1; exit }
    in_block { print }
    END { if (!seen || !complete) exit 1 }
  '); then
    printf 'missing PowerShell quickstart block in %s\n' "$readme" >&2
    exit 1
  fi

  if ! grep -Eq '^bash download-funasr-model\.sh sensevoice \./gguf([[:space:]]+#.*)?$' <<<"$quickstart"; then
    printf 'missing default SenseVoice download command in %s\n' "$readme" >&2
    exit 1
  fi
  if ! grep -Fx "./llama-funasr-sensevoice -m ./gguf/$DEFAULT_SENSEVOICE_MODEL --vad ./gguf/fsmn-vad.gguf -a audio.wav" <<<"$quickstart" >/dev/null; then
    printf 'README command does not run the default sensevoice-small-q8.gguf in %s\n' "$readme" >&2
    exit 1
  fi
  if ! grep -Fx "hf download FunAudioLLM/SenseVoiceSmall-GGUF $DEFAULT_SENSEVOICE_MODEL --local-dir .\\gguf" <<<"$powershell" >/dev/null; then
    printf 'missing Windows SenseVoice download command in %s\n' "$readme" >&2
    exit 1
  fi
  if ! grep -Fx 'hf download FunAudioLLM/fsmn-vad-GGUF fsmn-vad.gguf --local-dir .\gguf' <<<"$powershell" >/dev/null; then
    printf 'missing Windows FSMN-VAD download command in %s\n' "$readme" >&2
    exit 1
  fi
  if ! grep -Fx ".\\llama-funasr-sensevoice.exe -m .\\gguf\\$DEFAULT_SENSEVOICE_MODEL --vad .\\gguf\\fsmn-vad.gguf -a audio.wav" <<<"$powershell" >/dev/null; then
    printf 'missing Windows PowerShell quickstart in %s\n' "$readme" >&2
    exit 1
  fi
  if ! grep -F '[Releases](https://github.com/modelscope/FunASR/releases)' <<<"$section" >/dev/null; then
    printf 'non-portable Releases link in %s\n' "$readme" >&2
    exit 1
  fi
  if ! grep -F 'runtime-llamacpp-v0.1.9' <<<"$section" >/dev/null; then
    printf 'missing current runtime v0.1.9 link in %s\n' "$readme" >&2
    exit 1
  fi
  if ! grep -F 'funasr-llamacpp-linux-x64-vulkan' <<<"$section" >/dev/null; then
    printf 'missing Linux Vulkan runtime asset in %s\n' "$readme" >&2
    exit 1
  fi
  if ! grep -F -- '--backend vulkan' <<<"$section" >/dev/null; then
    printf 'missing Vulkan backend command hint in %s\n' "$readme" >&2
    exit 1
  fi
}

assert_readme_quickstart "$REPO_ROOT/README.md" '### CPU / Edge'
assert_readme_quickstart "$REPO_ROOT/README_zh.md" '### CPU / 边缘部署'

echo "download-funasr-model contract tests passed"
