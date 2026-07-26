#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <public-static-directory>" >&2
  exit 64
fi

SOURCE_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_ROOT="$(cd "${SCRIPT_DIR}/../product-site" && pwd)"
DEST_DIR="${SITE_ROOT}/legacy"
CONTENT_DIR="${SITE_ROOT}/content"
STAGE_DIR="$(mktemp -d "${SITE_ROOT}/.legacy-import.XXXXXX")"
MANIFEST_STAGE="$(mktemp "${SITE_ROOT}/.legacy-manifest.XXXXXX")"
PREVIOUS_DIR="${SITE_ROOT}/.legacy-previous.$$"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cleanup() {
  rm -rf "${STAGE_DIR}" "${MANIFEST_STAGE}"
  if [[ -d "${PREVIOUS_DIR}" && ! -d "${DEST_DIR}" ]]; then
    mv "${PREVIOUS_DIR}" "${DEST_DIR}"
  fi
}
trap cleanup EXIT

"${PYTHON_BIN}" - "${SOURCE_DIR}" "${STAGE_DIR}" "${MANIFEST_STAGE}" <<'PY'
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


source = Path(sys.argv[1]).resolve()
stage = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3]).resolve()

if not source.is_dir():
    raise SystemExit(f'public static directory does not exist: {source}')

allowed_suffixes = {
    '.html', '.css', '.js', '.json', '.xml', '.txt', '.ico', '.png',
    '.jpg', '.jpeg', '.webp', '.svg', '.woff', '.woff2', '.ttf',
    '.mp3', '.wav', '.ogg', '.m4a', '.mp4', '.webm',
}
forbidden_parts = {'stats', '.git', '.mcp-tasks', 'backup', 'backups', 'logs'}


def included(path: Path) -> bool:
    relative = path.relative_to(source)
    lowered_parts = {part.lower() for part in relative.parts}
    if forbidden_parts.intersection(lowered_parts):
        return False
    lowered_name = path.name.lower()
    if '.bak' in lowered_name or lowered_name.startswith('frontend-config-'):
        return False
    if lowered_name.startswith('.') or path.suffix.lower() == '.gz':
        return False
    return path.suffix.lower() in allowed_suffixes


selected = []
for path in sorted(source.rglob('*')):
    if path.is_symlink():
        raise SystemExit(f'symlink is not allowed in public corpus: {path}')
    if path.is_file() and included(path):
        selected.append(path)

if len(selected) < 100:
    raise SystemExit(f'public corpus unexpectedly small: {len(selected)} files')

hashes = {}
for source_path in selected:
    relative = source_path.relative_to(source)
    destination = stage / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    hashes[relative.as_posix()] = hashlib.sha256(destination.read_bytes()).hexdigest()

manifest = {
    'schema_version': 1,
    'source': 'https://www.funasr.com/ public static corpus',
    'captured': '2026-07-26',
    'files': dict(sorted(hashes.items())),
}
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
    encoding='utf-8',
)
print(f'imported {len(selected)} public files')
PY

mkdir -p "${CONTENT_DIR}"
if [[ -d "${DEST_DIR}" ]]; then
  mv "${DEST_DIR}" "${PREVIOUS_DIR}"
fi
mv "${STAGE_DIR}" "${DEST_DIR}"
mv "${MANIFEST_STAGE}" "${CONTENT_DIR}/legacy-manifest.json.tmp"
mv "${CONTENT_DIR}/legacy-manifest.json.tmp" "${CONTENT_DIR}/legacy-manifest.json"
rm -rf "${PREVIOUS_DIR}"
trap - EXIT
