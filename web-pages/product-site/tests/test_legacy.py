from __future__ import annotations

import hashlib
import json
import re
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
    assert soup.select_one('.nav-btn')['href'] == '/go/github'
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


def test_mlx_audio_ecosystem_entry_is_merged_attributed_and_bounded():
    pages = {
        'zh': LEGACY / 'ecosystem.html',
        'en': LEGACY / 'en' / 'ecosystem.html',
    }
    required_links = {
        'https://github.com/Blaizzy/mlx-audio',
        'https://github.com/Blaizzy/mlx-audio/blob/main/docs/models/stt/fun-asr-nano.md',
        'https://github.com/Blaizzy/mlx-audio/pull/885',
        '/go/fun-asr',
    }

    for language, path in pages.items():
        soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
        card = next(
            item
            for item in soup.select('.card')
            if item.select_one('.card-title').get_text(' ', strip=True) == 'MLX Audio'
        )
        text = card.get_text(' ', strip=True)
        hrefs = {link.get('href') for link in card.select('a[href]')}

        assert required_links <= hrefs
        assert 'Apple Silicon' in text
        assert 'Fun-ASR-Nano' in text
        assert 'OpenAI' in text
        assert ('仅转写' in text) if language == 'zh' else ('transcription-only' in text)
        assert ('无时间戳' in text) if language == 'zh' else ('no timestamps' in text)

    community_docs = {
        'en': Path(__file__).resolve().parents[3] / 'docs' / 'community_projects.md',
        'zh': Path(__file__).resolve().parents[3] / 'docs' / 'community_projects_zh.md',
    }
    for text_path in community_docs.values():
        text = text_path.read_text(encoding='utf-8')
        assert '[MLX Audio](https://github.com/Blaizzy/mlx-audio)' in text
        assert 'docs/models/stt/fun-asr-nano.md' in text
        assert 'https://github.com/Blaizzy/mlx-audio/pull/885' in text


def test_openmaic_ecosystem_entry_is_merged_attributed_and_bounded():
    pages = {
        'zh': LEGACY / 'ecosystem.html',
        'en': LEGACY / 'en' / 'ecosystem.html',
    }
    required_links = {
        'https://github.com/THU-MAIC/OpenMAIC',
        'https://github.com/THU-MAIC/OpenMAIC#funasr-local-asr',
        'https://github.com/THU-MAIC/OpenMAIC/pull/1044',
        'https://github.com/modelscope/FunASR',
    }

    for language, path in pages.items():
        soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
        cards = [
            item
            for item in soup.select('.card')
            if item.select_one('.card-title').get_text(' ', strip=True) == 'OpenMAIC'
        ]
        assert len(cards) == 1

        text = cards[0].get_text(' ', strip=True)
        hrefs = {link.get('href') for link in cards[0].select('a[href]')}

        assert required_links <= hrefs
        assert 'FunASR' in text
        assert 'SenseVoiceSmall' in text
        assert 'Paraformer' in text
        assert 'Fun-ASR-Nano' in text
        assert 'ASR_FUNASR_BASE_URL' in text
        assert ('本地' in text) if language == 'zh' else ('local' in text.lower())

    community_docs = {
        'en': Path(__file__).resolve().parents[3] / 'docs' / 'community_projects.md',
        'zh': Path(__file__).resolve().parents[3] / 'docs' / 'community_projects_zh.md',
    }
    for text_path in community_docs.values():
        text = text_path.read_text(encoding='utf-8')
        assert '[OpenMAIC](https://github.com/THU-MAIC/OpenMAIC)' in text
        assert 'ASR_FUNASR_BASE_URL' in text
        assert 'https://github.com/THU-MAIC/OpenMAIC/pull/1044' in text


def test_recent_merged_ecosystem_integrations_are_bilingual_and_attributed():
    pages = {
        'zh': LEGACY / 'ecosystem.html',
        'en': LEGACY / 'en' / 'ecosystem.html',
    }
    expected = {
        'RAGFlow': {
            'repo': 'https://github.com/infiniflow/ragflow',
            'pull': 'https://github.com/infiniflow/ragflow/pull/17388',
            'terms': ('FunASR', 'API key'),
        },
        'Omi': {
            'repo': 'https://github.com/BasedHardware/omi',
            'pull': 'https://github.com/BasedHardware/omi/pull/10447',
            'terms': ('Custom STT', 'raw audio'),
            'zh_terms': ('所选 STT provider', '转写文本和非音频数据仍可发送到 Omi'),
            'en_terms': ('selected STT provider', 'transcripts and non-audio data may still reach Omi'),
        },
        'UltraEval-Audio': {
            'repo': 'https://github.com/OpenBMB/UltraEval-Audio',
            'pull': 'https://github.com/OpenBMB/UltraEval-Audio/pull/47',
            'terms': ('Fun-ASR-Nano', 'revision'),
        },
        'GPT-SoVITS': {
            'repo': 'https://github.com/RVC-Boss/GPT-SoVITS',
            'pull': 'https://github.com/RVC-Boss/GPT-SoVITS/pull/2824',
            'terms': ('Fun-ASR-Nano', 'Transformers', '4.51', '<5'),
            'zh_terms': ('Qwen3', '转写'),
            'en_terms': ('Qwen3', 'transcription'),
        },
    }

    for language, path in pages.items():
        soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
        for name, contract in expected.items():
            cards = [
                card
                for card in soup.select('.card')
                if card.select_one('.card-title').get_text(' ', strip=True) == name
            ]
            assert len(cards) == 1
            text = cards[0].get_text(' ', strip=True)
            hrefs = {link.get('href') for link in cards[0].select('a[href]')}

            assert {contract['repo'], contract['pull']} <= hrefs
            assert all(term in text for term in contract['terms'])
            language_terms = contract.get(f'{language}_terms', ())
            assert all(term in text for term in language_terms)


def test_gpt_sovits_community_docs_track_the_merged_qwen3_dependency_fix():
    for relative in ('community_projects.md', 'community_projects_zh.md'):
        text = (Path(__file__).resolve().parents[3] / 'docs' / relative).read_text(
            encoding='utf-8'
        )
        assert 'https://github.com/RVC-Boss/GPT-SoVITS/pull/2824' in text
        assert 'Transformers' in text
        assert '4.51' in text
        assert '<5' in text


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


def test_v145_release_pages_are_bilingual_indexed_and_operational():
    slug = 'funasr-v1-4-5-pypi-llama-cpp-release.html'
    pages = {
        'zh': LEGACY / 'blog' / slug,
        'en': LEGACY / 'en' / 'blog' / slug,
    }
    release_assets = (
        'funasr-1.4.5-py3-none-any.whl',
        'funasr-1.4.5.tar.gz',
        'funasr-llamacpp-linux-arm64.tar.gz',
        'funasr-llamacpp-linux-x64-avx2.tar.gz',
        'funasr-llamacpp-linux-x64-vulkan.tar.gz',
        'funasr-llamacpp-linux-x64.tar.gz',
        'funasr-llamacpp-macos-arm64.tar.gz',
        'funasr-llamacpp-windows-x64-avx2.zip',
        'funasr-llamacpp-windows-x64-cuda.zip',
        'funasr-llamacpp-windows-x64-vulkan.zip',
        'funasr-llamacpp-windows-x64.zip',
        'SHA256SUMS-v1.4.5',
    )
    language_contracts = {
        'zh': ('不再是硬依赖', '签名标签', '不代表其他设备或生产并发容量'),
        'en': ('no longer a hard dependency', 'signed tag', 'not a production capacity promise'),
    }

    for language, path in pages.items():
        text = path.read_text(encoding='utf-8')
        soup = BeautifulSoup(text, 'html.parser')

        assert 'funasr==1.4.5' in text
        assert 'funasr[knf]==1.4.5' in text
        assert 'https://pypi.org/project/funasr/1.4.5/' in text
        assert 'https://github.com/modelscope/FunASR/releases/tag/v1.4.5' in text
        assert 'runtime-llamacpp-v0.2.1' in text
        assert 'SHA256SUMS-v1.4.5' in text
        assert 'torchaudio' in text
        assert 'Ascend 910B' in text
        assert all(value in text for value in ('70.47', '1.15', '0.016'))
        assert '4df59cc15386ff3bb10916256d807ebc5c85f81d' in text
        assert all(marker in text for marker in language_contracts[language])

        asset_codes = {
            code.get_text(strip=True)
            for code in soup.select('code')
            if code.get_text(strip=True).startswith(('funasr-', 'SHA256SUMS-'))
        }
        assert asset_codes == set(release_assets)

        expected_route = f'/{"" if language == "zh" else "en/"}blog/{slug}'
        assert soup.select_one('link[rel="canonical"]')['href'].endswith(expected_route)
        peer_route = f'/{"en/" if language == "zh" else ""}blog/{slug}'
        peer_language = 'en' if language == 'zh' else 'zh'
        assert soup.select_one(
            f'link[rel="alternate"][hreflang="{peer_language}"]'
            f'[href="https://www.funasr.com{peer_route}"]'
        )
        assert soup.select_one(
            'link[rel="alternate"][hreflang="x-default"]'
            f'[href="https://www.funasr.com/en/blog/{slug}"]'
        )
        metadata = json.loads(soup.select_one('script[type="application/ld+json"]').string)
        assert metadata['datePublished'] == '2026-08-28'
        assert metadata['dateModified'] == '2026-08-28'

    zh_index = (LEGACY / 'blog' / 'index.html').read_text(encoding='utf-8')
    en_index = (LEGACY / 'en' / 'blog' / 'index.html').read_text(encoding='utf-8')
    sitemap = (LEGACY / 'sitemap.xml').read_text(encoding='utf-8')

    assert f'/blog/{slug}' in zh_index
    assert f'/en/blog/{slug}' in en_index
    assert f'https://www.funasr.com/blog/{slug}' in sitemap
    assert f'https://www.funasr.com/en/blog/{slug}' in sitemap


def test_llama_cpp_blog_points_directly_to_current_runtime_release():
    pages = {
        'zh': LEGACY / 'blog' / 'funasr-llama-cpp-whisper-cpp-alternative.html',
        'en': LEGACY / 'en' / 'blog' / 'funasr-llama-cpp-whisper-cpp-alternative.html',
    }
    expected_routes = {
        'zh': '/deploy/llama-cpp.html',
        'en': '/en/deploy/llama-cpp.html',
    }

    for language, path in pages.items():
        text = path.read_text(encoding='utf-8')
        soup = BeautifulSoup(text, 'html.parser')
        hrefs = {link.get('href') for link in soup.select('a[href]')}

        assert 'runtime-llamacpp-v0.2.0' in text
        assert 'runtime-llamacpp-v0.1.9' not in text
        assert expected_routes[language] in hrefs
        assert '2026-08-15' in soup.select_one('script[type="application/ld+json"]').string
        assert '**' not in soup.select_one('article').get_text()


def test_public_pages_do_not_overstate_sensevoice_language_or_speed_claims():
    language_claim = re.compile(
        r'50\+\s*(?:supported\s+|支持\s*)?(?:languages?|语言|语种)',
        re.IGNORECASE,
    )
    fixed_speed_claim = re.compile(
        r'\b(?:13|15|17|170)[x×]\b|(?:13|15|17|170)\s*倍'
    )

    violations = []
    for path in sorted(LEGACY.rglob('*.html')):
        source = path.read_text(encoding='utf-8')
        text = BeautifulSoup(source, 'html.parser').get_text(
            ' ',
            strip=True,
        )
        searchable = f'{source}\n{text}'
        if language_claim.search(searchable) or fixed_speed_claim.search(searchable):
            violations.append(path.relative_to(LEGACY).as_posix())

    assert violations == []


def test_funclip_v220_moss_release_pages_are_bilingual_indexed_and_verifiable():
    slug = 'funclip-v2-2-0-moss-speaker-clipping.html'
    pages = {
        'zh': LEGACY / 'blog' / slug,
        'en': LEGACY / 'en' / 'blog' / slug,
    }
    language_contracts = {
        'zh': ('第三方模型', '段级时间戳', '不接外部 VAD 或说话人模型'),
        'en': ('third-party model', 'segment-level timestamps', 'no external VAD or speaker model'),
    }

    for language, path in pages.items():
        text = path.read_text(encoding='utf-8')
        soup = BeautifulSoup(text, 'html.parser')
        hrefs = {link.get('href') for link in soup.select('a[href]')}

        assert 'FunClip v2.2.0' in text
        assert 'OpenMOSS-Team/MOSS-Transcribe-Diarize' in text
        assert 'e8681d68e7042738ffca8ac8212bc8fcb1131ab8' in text
        assert '/v1/audio/transcriptions' in text
        assert 'response_format=json' in text
        assert '994c5d9cf392b74b36284d526eca8bada1560a3e7825ab7baa9c673a1b4ef216' in text
        assert '4f5a7d33d9ea65467f29b55b15ed5be18e64de7e57e2f9f36fe51a23b40557e7' in text
        assert all(marker in text for marker in language_contracts[language])
        assert {
            'https://github.com/modelscope/FunClip/releases/tag/v2.2.0',
            '/deploy/moss-transcribe-diarize.html'
            if language == 'zh'
            else '/en/deploy/moss-transcribe-diarize.html',
        } <= hrefs

        route = f'/{"" if language == "zh" else "en/"}blog/{slug}'
        peer = f'/{"en/" if language == "zh" else ""}blog/{slug}'
        assert soup.select_one('link[rel="canonical"]')['href'].endswith(route)
        assert soup.select_one(f'link[rel="alternate"][href$="{peer}"]')
        image = soup.select_one('article img[src="/img/funclip-v2-1-0-interface.jpg"]')
        assert image
        metadata = json.loads(soup.select_one('script[type="application/ld+json"]').string)
        assert metadata['datePublished'] == '2026-08-31'
        assert metadata['dateModified'] == '2026-08-31'

    zh_index = (LEGACY / 'blog' / 'index.html').read_text(encoding='utf-8')
    en_index = (LEGACY / 'en' / 'blog' / 'index.html').read_text(encoding='utf-8')
    sitemap = (LEGACY / 'sitemap.xml').read_text(encoding='utf-8')
    assert f'/blog/{slug}' in zh_index
    assert f'/en/blog/{slug}' in en_index
    assert f'https://www.funasr.com/blog/{slug}' in sitemap
    assert f'https://www.funasr.com/en/blog/{slug}' in sitemap
