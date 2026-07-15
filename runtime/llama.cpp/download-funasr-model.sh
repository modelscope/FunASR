#!/usr/bin/env bash
# Download pre-converted FunASR GGUF models from Hugging Face in one command.
#   ./download-funasr-model.sh sensevoice [outdir] [q8|f16|f32|all]
#   ./download-funasr-model.sh paraformer [outdir] [q8|f16|f32|all]
#   ./download-funasr-model.sh nano [outdir] [q8_0|q4km|q5km|all]
#   ./download-funasr-model.sh fsmn-vad [outdir]
# ASR models also pull fsmn-vad.gguf (needed for built-in --vad long-audio segmentation).
set -euo pipefail
usage(){
  cat >&2 <<EOF
usage: $0 {sensevoice|paraformer|nano|fsmn-vad} [outdir] [variant]
  sensevoice/paraformer variants: q8 (default), f16, f32, all
  nano variants: q8_0 (default), q4km, q5km, all
EOF
  exit 1
}
[ $# -ge 1 ] && [ $# -le 3 ] || usage
MODEL="$1"; OUT="${2:-funasr-gguf}"; VARIANT="${3:-}"
case "$MODEL" in
  sensevoice)
    REPO="FunAudioLLM/SenseVoiceSmall-GGUF"
    case "${VARIANT:-q8}" in
      q8)  FILES=(sensevoice-small-q8.gguf) ;;
      f16) FILES=(sensevoice-small-f16.gguf) ;;
      f32) FILES=(sensevoice-small.gguf) ;;
      all) FILES=() ;;
      *) usage ;;
    esac
    ;;
  paraformer)
    REPO="FunAudioLLM/Paraformer-GGUF"
    case "${VARIANT:-q8}" in
      q8)  FILES=(paraformer-q8.gguf) ;;
      f16) FILES=(paraformer-f16.gguf) ;;
      f32) FILES=(paraformer.gguf) ;;
      all) FILES=() ;;
      *) usage ;;
    esac
    ;;
  nano)
    REPO="FunAudioLLM/Fun-ASR-Nano-GGUF"
    case "${VARIANT:-q8_0}" in
      q8|q8_0) FILES=(funasr-encoder-f16.gguf qwen3-0.6b-q8_0.gguf) ;;
      q4|q4km) FILES=(funasr-encoder-f16.gguf qwen3-0.6b-q4km.gguf) ;;
      q5|q5km) FILES=(funasr-encoder-f16.gguf qwen3-0.6b-q5km.gguf) ;;
      all) FILES=() ;;
      *) usage ;;
    esac
    ;;
  fsmn-vad|vad)
    [ -z "$VARIANT" ] || usage
    REPO="FunAudioLLM/fsmn-vad-GGUF"
    FILES=(fsmn-vad.gguf)
    ;;
  *) usage ;;
esac
# huggingface_hub ships `hf` (new CLI); older versions only have `huggingface-cli` (deprecated). Use whichever exists.
if   command -v hf              >/dev/null 2>&1; then HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
else echo "need the Hugging Face CLI: pip install -U huggingface_hub"; exit 1; fi
mkdir -p "$OUT"
echo "downloading $REPO ..."
if [ ${#FILES[@]} -eq 0 ]; then
  "$HF" download "$REPO" --include "*.gguf" --local-dir "$OUT"
else
  "$HF" download "$REPO" "${FILES[@]}" --local-dir "$OUT"
fi
if [ "$MODEL" != "fsmn-vad" ] && [ "$MODEL" != "vad" ]; then
  echo "downloading FSMN-VAD (for --vad) ..."
  "$HF" download FunAudioLLM/fsmn-vad-GGUF fsmn-vad.gguf --local-dir "$OUT"
fi
echo "done -> $OUT"; ls -1 "$OUT"/*.gguf
