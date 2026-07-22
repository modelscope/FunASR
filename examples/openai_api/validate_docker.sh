#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash validate_docker.sh [--gpu] [--tag funasr-api:local] [--port 8000]

Build and smoke-test the OpenAI-compatible FunASR Docker example.

Default mode validates the portable CPU image:
  bash validate_docker.sh

GPU mode requires NVIDIA Container Toolkit and a CUDA-capable image:
  bash validate_docker.sh --gpu
USAGE
}

want_gpu=0
image_tag="funasr-api:local"
host_port="8000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      want_gpu=1
      shift
      ;;
    --tag)
      image_tag="${2:?missing value for --tag}"
      shift 2
      ;;
    --port)
      host_port="${2:?missing value for --port}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker CLI is required for Docker validation." >&2
  exit 127
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

container_name="funasr-api-validate-${host_port}"
base_url="http://127.0.0.1:${host_port}"

cleanup() {
  docker rm -f "$container_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "$want_gpu" -eq 1 ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for --gpu validation." >&2
    exit 127
  fi
  if ! docker info 2>/dev/null | grep -qi "nvidia"; then
    echo "NVIDIA Container Toolkit is required for --gpu validation." >&2
    exit 1
  fi
fi

echo "Building ${image_tag} from examples/openai_api/Dockerfile"
docker build -t "$image_tag" .

cleanup

docker_args=(docker run --rm -d --name "$container_name" -p "${host_port}:8000")
if [[ "$want_gpu" -eq 1 ]]; then
  docker_args+=(--gpus all -e FUNASR_DEVICE=cuda)
else
  docker_args+=(-e FUNASR_DEVICE=cpu)
fi
docker_args+=(-e FUNASR_MODEL=sensevoice "$image_tag")

echo "Starting ${container_name}"
"${docker_args[@]}"

echo "Waiting for /health"
for _ in $(seq 1 60); do
  if python - "$base_url" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

urllib.request.urlopen(sys.argv[1] + "/health", timeout=2).read()
PY
  then
    break
  fi
  sleep 2
done

python smoke_test.py --base-url "$base_url"

if [[ "$want_gpu" -eq 1 ]]; then
  echo "GPU Docker validation passed for ${image_tag}"
else
  echo "CPU Docker validation passed for ${image_tag}"
fi
