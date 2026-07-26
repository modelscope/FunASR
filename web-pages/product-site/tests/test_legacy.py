from __future__ import annotations

import hashlib
import json
from pathlib import Path


SITE_ROOT = Path(__file__).resolve().parents[1]
LEGACY = SITE_ROOT / 'legacy'
MANIFEST = SITE_ROOT / 'content' / 'legacy-manifest.json'


def public_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob('*') if path.is_file())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def test_legacy_manifest_matches_snapshot():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    actual = {
        path.relative_to(LEGACY).as_posix(): sha256(path)
        for path in public_files(LEGACY)
    }

    assert actual == manifest['files']
    assert manifest['captured'] == '2026-07-26'
    assert len(actual) >= 100


def test_private_or_generated_files_are_excluded():
    forbidden_parts = {'access.log', 'stats', '.git', '.mcp-tasks', 'backup', 'backups'}

    for path in public_files(LEGACY):
        relative = path.relative_to(LEGACY)
        assert not forbidden_parts.intersection(relative.parts)
        assert '.bak' not in path.name
        assert not path.name.startswith('frontend-config-')
        assert path.suffix != '.gz'


def test_required_legacy_contracts_are_preserved():
    required = {
        'index.html',
        'en/index.html',
        'blog/index.html',
        'en/blog/index.html',
        'donors.html',
        'en/donors.html',
        'llama-cpp.html',
        'en/llama-cpp.html',
        'static/offline/index.html',
        'static/online/index.html',
        'voice/index.html',
        'robots.txt',
        'sitemap.xml',
    }
    actual = {
        path.relative_to(LEGACY).as_posix()
        for path in public_files(LEGACY)
    }

    assert required <= actual
