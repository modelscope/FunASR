from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from bs4 import BeautifulSoup


SITE_ROOT = Path(__file__).resolve().parents[1]
LEGACY = SITE_ROOT / 'legacy'
MANIFEST = SITE_ROOT / 'content' / 'legacy-manifest.json'
sys.path.insert(0, str(SITE_ROOT))

from legacy import normalize_document  # noqa: E402


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


def test_navigation_normalization_is_idempotent_and_preserves_article():
    article = '<article><p>Keep <code>this &amp; that</code> exactly.</p></article>'
    sample = f'''<!doctype html>
<html lang="zh"><head>
<link rel="canonical" href="https://www.funasr.com/blog/example.html">
<link rel="alternate" hreflang="en" href="https://www.funasr.com/en/blog/example.html">
<link href="https://fonts.googleapis.com/css2?family=Inter" rel="stylesheet">
</head><body>
<nav class="nav"><div>Old navigation</div></nav>
{article}
<script src="/stats/tracker.js"></script>
</body></html>'''

    once = normalize_document(sample, '/blog/example.html', 'zh')
    twice = normalize_document(once, '/blog/example.html', 'zh')
    soup = BeautifulSoup(once, 'html.parser')

    assert once == twice
    assert article in once
    assert [link.get_text(strip=True) for link in soup.select('.nav-links a')][-1] == '功德榜'
    assert soup.select_one('link[rel="canonical"]')['href'].endswith('/blog/example.html')
    assert not soup.select_one('link[href*="fonts.googleapis.com"]')
    assert not soup.select_one('script[src="/stats/tracker.js"]')


def test_english_navigation_uses_language_peer():
    sample = '<html><head></head><body><nav class="nav"></nav><article>Body</article></body></html>'
    soup = BeautifulSoup(
        normalize_document(sample, '/en/blog/example.html', 'en'),
        'html.parser',
    )

    assert [link.get_text(strip=True) for link in soup.select('.nav-links a')][-1] == 'Donors'
    assert soup.select_one('[data-language-peer]')['href'] == '/blog/example.html'


def test_v140_release_pages_are_bilingual_indexed_and_precise():
    slug = 'funasr-v1-4-0-pypi-release.html'
    pages = {
        'zh': LEGACY / 'blog' / slug,
        'en': LEGACY / 'en' / 'blog' / slug,
    }
    runtime_assets = (
        'linux-arm64',
        'linux-x64',
        'linux-x64-avx2',
        'linux-x64-vulkan',
        'macos-arm64',
        'windows-x64',
        'windows-x64-avx2',
        'windows-x64-cuda',
        'windows-x64-vulkan',
    )

    for language, path in pages.items():
        text = path.read_text(encoding='utf-8')
        soup = BeautifulSoup(text, 'html.parser')

        assert 'funasr==1.4.0' in text
        assert 'https://github.com/modelscope/FunASR/releases/tag/v1.4.0' in text
        assert 'SHA256SUMS-v1.4.0' in text
        assert '`vda_model`' not in text
        assert 'vda_model' in text
        assert 'vad_model' in text
        assert 'SenseVoice' in text
        assert 'RWKV-BAT' in text
        assert 'PyPI wheel' in text
        assert 'tagged source' in text
        assert all(asset in text for asset in runtime_assets)

        expected_route = f'/{"" if language == "zh" else "en/"}blog/{slug}'
        assert soup.select_one('link[rel="canonical"]')['href'].endswith(expected_route)

    zh_index = (LEGACY / 'blog' / 'index.html').read_text(encoding='utf-8')
    en_index = (LEGACY / 'en' / 'blog' / 'index.html').read_text(encoding='utf-8')
    sitemap = (LEGACY / 'sitemap.xml').read_text(encoding='utf-8')

    assert f'/blog/{slug}' in zh_index
    assert f'/en/blog/{slug}' in en_index
    assert f'https://www.funasr.com/blog/{slug}' in sitemap
    assert f'https://www.funasr.com/en/blog/{slug}' in sitemap
