#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-sensevoice}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-verbose_json}"
AUDIO_PATH="${1:-sample.wav}"
SAMPLE_URL="${SAMPLE_URL:-https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav}"

if [ ! -f "$AUDIO_PATH" ]; then
  echo "Downloading sample audio to $AUDIO_PATH"
  curl -L "$SAMPLE_URL" -o "$AUDIO_PATH"
fi

echo "Checking $BASE_URL/health"
curl -fsS "$BASE_URL/health"
printf '\n\n'

echo "Transcribing $AUDIO_PATH with model=$MODEL"
curl -fsS "$BASE_URL/v1/audio/transcriptions" \
  -F "file=@${AUDIO_PATH}" \
  -F "model=${MODEL}" \
  -F "response_format=${RESPONSE_FORMAT}"
printf '\n'
