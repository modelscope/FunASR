#!/usr/bin/env bash
# Download pre-converted FunASR GGUF models from Hugging Face — one command, no Python ML env.
#   ./download-funasr-model.sh sensevoice [outdir]
#   ./download-funasr-model.sh paraformer | nano | fsmn-vad
# ASR models also pull fsmn-vad.gguf (needed for built-in --vad long-audio segmentation).
set -euo pipefail
usage(){ echo "usage: $0 {sensevoice|paraformer|nano|fsmn-vad} [outdir]"; exit 1; }
[ $# -ge 1 ] || usage
MODEL="$1"; OUT="${2:-funasr-gguf}"
case "$MODEL" in
  sensevoice) REPO="FunAudioLLM/SenseVoiceSmall-GGUF" ;;
  paraformer) REPO="FunAudioLLM/Paraformer-GGUF" ;;
  nano)       REPO="FunAudioLLM/Fun-ASR-Nano-GGUF" ;;
  fsmn-vad|vad) REPO="FunAudioLLM/fsmn-vad-GGUF" ;;
  *) usage ;;
esac
# huggingface_hub ships `hf` (new CLI); older versions only have `huggingface-cli` (deprecated). Use whichever exists.
if   command -v hf              >/dev/null 2>&1; then HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
else echo "need the Hugging Face CLI: pip install -U huggingface_hub"; exit 1; fi
mkdir -p "$OUT"
echo "downloading $REPO ..."; "$HF" download "$REPO" --include "*.gguf" --local-dir "$OUT"
if [ "$MODEL" != "fsmn-vad" ] && [ "$MODEL" != "vad" ]; then
  echo "downloading FSMN-VAD (for --vad) ..."; "$HF" download FunAudioLLM/fsmn-vad-GGUF --include "*.gguf" --local-dir "$OUT"
fi
echo "done -> $OUT"; ls -1 "$OUT"/*.gguf
