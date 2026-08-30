from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from bs4 import BeautifulSoup


SITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE_ROOT))

from build import build  # noqa: E402
from registry import BENCHMARK_FIELDS, load_registry  # noqa: E402
from validate import validate_output  # noqa: E402


@pytest.fixture
def built_site(tmp_path):
    build(tmp_path)
    return tmp_path


def read_soup(path: Path) -> BeautifulSoup:
    return BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')


def route_path(root: Path, route: str) -> Path:
    if route.endswith('/'):
        return root / route.lstrip('/') / 'index.html'
    return root / route.lstrip('/')


@pytest.mark.parametrize(
    ('relative', 'markers'),
    (
        (
            'index.html',
            ('工业语音工具箱', '高精度转写', '多语种与情感', '智能视频剪辑'),
        ),
        (
            'en/index.html',
            ('Production speech toolkit', 'High-accuracy ASR', 'Languages and emotion', 'AI video editing'),
        ),
    ),
)
def test_homepage_routes_each_project_by_workload(built_site, relative, markers):
    soup = read_soup(built_site / relative)
    section = soup.select_one('#projects')

    assert section
    rows = section.select('[data-project]')
    assert [row['data-project'] for row in rows] == [
        'funasr',
        'fun-asr',
        'sensevoice',
        'funclip',
    ]
    assert {link['href'] for link in section.select('a[href]')} == {
        '/go/github',
        '/go/fun-asr',
        '/go/sensevoice',
        '/go/funclip',
    }
    text = section.get_text(' ', strip=True)
    for marker in markers:
        assert marker in text


def test_every_deployment_page_has_operational_contract(built_site):
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')

    for entry in registry['deployments']:
        for language in ('zh', 'en'):
            page = route_path(built_site, entry['routes'][language])
            soup = read_soup(page)
            assert soup.select_one('[data-field="verified-date"]')
            assert soup.select_one('[data-section="fit"]')
            assert soup.select_one('[data-section="commands"]')
            assert soup.select_one('[data-section="smoke-test"]')
            assert soup.select_one('[data-section="security"]')
            assert soup.select_one('[data-section="limitations"]')
            assert soup.select_one('[data-section="operations"]')
            assert soup.select_one('[data-section="evidence"]')


def test_deployment_pages_publish_accurate_discovery_metadata(built_site):
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')

    for entry in registry['deployments']:
        for language in ('zh', 'en'):
            page = route_path(built_site, entry['routes'][language])
            soup = read_soup(page)
            metadata = json.loads(soup.select_one('script[type="application/ld+json"]').string)

            assert metadata['license'] == (
                'https://github.com/modelscope/FunASR/blob/main/LICENSE'
            )
            assert metadata['dateModified'] == entry['tested']['verified']


def test_deployment_sitemap_entries_use_registry_verification_dates(built_site):
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')
    root = ET.parse(built_site / 'sitemap.xml').getroot()
    namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    entries = {
        item.findtext('sitemap:loc', namespaces=namespace): item.findtext(
            'sitemap:lastmod', namespaces=namespace
        )
        for item in root.findall('sitemap:url', namespace)
    }

    for entry in registry['deployments']:
        for language in ('zh', 'en'):
            url = f"https://www.funasr.com{entry['routes'][language]}"
            assert entries[url] == entry['tested']['verified']


def test_detail_commands_come_from_registry(built_site):
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')

    for entry in registry['deployments']:
        soup = read_soup(route_path(built_site, entry['routes']['en']))
        rendered = soup.get_text('\n')
        for command in entry['commands']['smoke']:
            assert command in rendered


def test_moss_detail_renders_separate_vllm_and_sglang_runtime_paths(built_site):
    for relative in (
        'deploy/moss-transcribe-diarize.html',
        'en/deploy/moss-transcribe-diarize.html',
    ):
        soup = read_soup(built_site / relative)
        paths = soup.select('[data-runtime-path]')

        assert [path['data-runtime-path'] for path in paths] == ['vllm', 'sglang-omni']
        rendered = '\n'.join(path.get_text('\n') for path in paths)
        assert 'vllm serve OpenMOSS-Team/MOSS-Transcribe-Diarize' in rendered
        assert 'sgl-omni serve' in rendered
        assert 'response_format=diarized_json' in rendered
        assert 'response_format=verbose_json' in rendered


def test_realtime_page_publishes_verified_v142_quickstart(built_site):
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')
    entry = next(item for item in registry['deployments'] if item['id'] == 'realtime')
    commands = '\n'.join(
        command
        for group in ('install', 'launch', 'smoke')
        for command in entry['commands'][group]
    )

    assert entry['tested']['funasr'] == '1.4.2'
    assert entry['tested']['verified'] == '2026-08-17'
    assert 'git clone --branch v1.4.2 --depth 1' in commands
    assert 'runtime/python/websocket/requirements_server.txt' in commands
    assert 'cd FunASR/runtime/python/websocket && python funasr_wss_server.py' in commands
    assert '../../funasr_api/asr_example.wav' in commands
    assert 'tests/test_audio/zh.wav' not in commands

    for relative in ('deploy/realtime.html', 'en/deploy/realtime.html'):
        soup = read_soup(built_site / relative)
        source = soup.select_one('.detail-actions a[href="/go/github"]')
        assert source


@pytest.mark.parametrize(
    ('relative', 'boundary'),
    (
        ('deploy/llama-cpp.html', 'Windows AMD'),
        ('en/deploy/llama-cpp.html', 'Windows AMD'),
    ),
)
def test_llama_cpp_pages_render_v026_download_matrix(built_site, relative, boundary):
    soup = read_soup(built_site / relative)
    section = soup.select_one('[data-section="downloads"]')

    assert section
    rows = section.select('[data-download-asset]')
    assert len(rows) == 10
    assert all(row.select_one('a[href*="runtime-llamacpp-v0.2.6"]') for row in rows)
    assert any('cuda-blackwell' in row.get_text(' ', strip=True).lower() for row in rows)
    assert all(len(row.select_one('[data-field="sha256"]').get_text(strip=True)) == 64 for row in rows)
    assert boundary in soup.get_text(' ', strip=True)
    text = soup.get_text(' ', strip=True)
    assert 'F16' in text
    assert 'initializing' in text
    assert 'resolving buffer type' in text
    assert 'backend ready' in text
    assert 'model ready' in text
    assert 'graph allocated' in text
    assert 'compute starting' in text


@pytest.mark.parametrize('relative', ('benchmarks.html', 'en/benchmarks.html'))
def test_benchmark_page_surfaces_v024_f16_stability(built_site, relative):
    soup = read_soup(built_site / relative)
    record = next(
        row
        for row in soup.select('[data-benchmark-record]')
        if 'SenseVoiceSmall F16 GGUF' in row.get_text(' ', strip=True)
    )
    text = record.get_text(' ', strip=True)

    assert 'runtime-llamacpp-v0.2.4' in text
    assert '100/100' in text
    assert '0 empty transcripts' in text
    assert 'not an accuracy study or production capacity promise' in text
    assert record.select_one(
        'a[href="https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.2.4"]'
    )


@pytest.mark.parametrize(
    ('relative', 'language_marker'),
    (
        ('deploy/sensevoice-native-server.html', '连接上限'),
        ('en/deploy/sensevoice-native-server.html', 'connection limit'),
    ),
)
def test_sensevoice_native_server_pages_render_operational_contract(
    built_site, relative, language_marker
):
    soup = read_soup(built_site / relative)
    text = soup.get_text(' ', strip=True)

    for marker in (
        'sensevoice-server',
        '/v1/audio/transcriptions',
        '/v1/realtime?intent=transcription',
        '--max-connections',
        '--max-audio-seconds',
        language_marker,
    ):
        assert marker in text


def test_benchmark_rows_have_complete_conditions(built_site):
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')
    records = [record for entry in registry['deployments'] for record in entry['benchmarks']]
    soup = read_soup(built_site / 'en/benchmarks.html')
    rows = soup.select('[data-benchmark-record]')

    assert len(records) >= 3
    assert len(rows) == len(records)
    for record in records:
        assert all(record.get(field) for field in BENCHMARK_FIELDS)
    for row in rows:
        assert row.select_one('[data-field="benchmark-hardware"]')
        assert row.select_one('[data-field="benchmark-settings"]')
        assert row.select_one('[data-field="benchmark-timing"]')
        assert row.select_one('[data-field="benchmark-source"]')
        assert row.select_one('[data-field="benchmark-qualification"]')


def test_benchmark_page_warns_against_cross_profile_claims(built_site):
    for relative in ('benchmarks.html', 'en/benchmarks.html'):
        soup = read_soup(built_site / relative)
        warning = soup.select_one('[data-benchmark-warning]')
        assert warning
        assert 'RTFx' in warning.get_text()


@pytest.mark.parametrize(
    ('relative', 'release_caveat'),
    (
        ('benchmarks.html', '不代表 SGLang-Omni 已发布该集成'),
        ('en/benchmarks.html', 'not a released SGLang-Omni integration'),
    ),
)
def test_rtx4090_community_benchmark_is_bilingual_and_qualified(
    built_site, relative, release_caveat
):
    soup = read_soup(built_site / relative)
    section = soup.select_one('[data-community-benchmark="rtx4090"]')

    assert section
    text = section.get_text(' ', strip=True)
    for marker in ('105,067', '16.11 GiB', '0.0175', '0.0164', release_caveat):
        assert marker in text

    links = {link.get('href') for link in section.select('a[href]')}
    assert {
        'https://github.com/sgl-project/sglang-omni/issues/1170',
        'https://github.com/sgl-project/sglang-omni/issues/1120',
        'https://github.com/sgl-project/sglang-omni/pull/1171',
        'https://gist.github.com/wirybeaver/ffa8a07f89066654a271a45b21592d25',
    } <= links


def test_manifest_contains_all_detail_and_benchmark_routes(built_site):
    manifest = __import__('json').loads(
        (built_site / 'deployment-manifest.json').read_text(encoding='utf-8')
    )
    routes = {page['route'] for page in manifest['pages']}
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')
    expected = {
        entry['routes'][language]
        for entry in registry['deployments']
        for language in ('zh', 'en')
    }
    expected.update({'/benchmarks.html', '/en/benchmarks.html'})

    assert expected <= routes


def test_old_llama_routes_point_to_product_pages(built_site):
    for relative, expected in (
        ('llama-cpp.html', '/deploy/llama-cpp.html'),
        ('en/llama-cpp.html', '/en/deploy/llama-cpp.html'),
    ):
        soup = read_soup(built_site / relative)
        assert soup.select_one('link[rel="canonical"]')['href'].endswith(expected)
        assert soup.select_one('.nav-links a[href$="/deploy/"]') or soup.select_one(
            '.nav-links a[href$="/en/deploy/"]'
        )


@pytest.mark.parametrize('relative', ('models.html', 'en/models.html'))
def test_model_pages_use_attributed_repository_routes(built_site, relative):
    soup = read_soup(built_site / relative)
    hrefs = {link.get('href') for link in soup.select('a[href]')}

    assert '/go/fun-asr' in hrefs
    assert '/go/sensevoice' in hrefs
    assert 'https://github.com/QwenAudio/Fun-ASR' not in hrefs
    assert 'https://github.com/QwenAudio/SenseVoice' not in hrefs


@pytest.mark.parametrize('relative', ('ecosystem.html', 'en/ecosystem.html'))
def test_ecosystem_pages_use_attributed_funclip_route(built_site, relative):
    soup = read_soup(built_site / relative)
    hrefs = {link.get('href') for link in soup.select('a[href]')}

    assert '/go/funclip' in hrefs
    assert 'https://github.com/modelscope/FunClip' not in hrefs


@pytest.mark.parametrize('relative', ('ecosystem.html', 'en/ecosystem.html'))
def test_ecosystem_surfaces_orca_sensevoice_desktop_integration(built_site, relative):
    soup = read_soup(built_site / relative)
    anchor = soup.select_one(
        '.card-title a[href="https://github.com/stablyai/orca"]'
    )

    assert anchor
    card = anchor.find_parent(class_='card')
    assert card
    links = {link.get('href') for link in card.select('a[href]')}
    assert 'https://github.com/stablyai/orca/pull/7436' in links
    assert 'https://github.com/stablyai/orca/releases/tag/v1.4.159-rc.1' in links
    text = card.get_text(' ', strip=True)
    for marker in ('SenseVoice', 'macOS', 'Linux', 'Windows', 'v1.4.158'):
        assert marker in text


@pytest.mark.parametrize('relative', ('ecosystem.html', 'en/ecosystem.html'))
def test_ecosystem_surfaces_merged_whisperlivekit_sensevoice_backend(
    built_site, relative
):
    soup = read_soup(built_site / relative)
    anchor = soup.select_one(
        '.card-title a[href="https://github.com/QuentinFuxa/WhisperLiveKit"]'
    )

    assert anchor
    card = anchor.find_parent(class_='card')
    assert card
    links = {link.get('href') for link in card.select('a[href]')}
    assert 'https://github.com/QuentinFuxa/WhisperLiveKit/pull/385' in links
    text = card.get_text(' ', strip=True)
    for marker in ('SenseVoiceSmall', 'LocalAgreement', 'VAC/VAD', 'timestamp'):
        assert marker in text


@pytest.mark.parametrize('relative', ('ecosystem.html', 'en/ecosystem.html'))
def test_ecosystem_surfaces_released_subtitle_edit_funasr_backends(
    built_site, relative
):
    soup = read_soup(built_site / relative)
    anchor = soup.select_one(
        '.card-title a[href="https://github.com/SubtitleEdit/subtitleedit"]'
    )

    assert anchor
    card = anchor.find_parent(class_='card')
    assert card
    links = {link.get('href') for link in card.select('a[href]')}
    assert {
        'https://github.com/SubtitleEdit/subtitleedit/pull/13063',
        'https://github.com/SubtitleEdit/subtitleedit/releases/tag/v5.2.0-beta2',
    } <= links
    text = card.get_text(' ', strip=True)
    for marker in (
        '13.7K',
        'Fun-ASR Nano',
        'SenseVoice',
        'Q4',
        'Q8',
        'F16',
        'Windows',
        'macOS',
        'Linux',
        'v5.2.0-beta2',
    ):
        assert marker in text


@pytest.mark.parametrize('relative', ('ecosystem.html', 'en/ecosystem.html'))
def test_ecosystem_refresh_tracks_current_release_and_merged_native_runtime(
    built_site, relative
):
    soup = read_soup(built_site / relative)
    text = soup.get_text(' ', strip=True)

    assert '36K+' in text
    assert soup.select_one(
        'a[href="https://github.com/modelscope/FunClip/releases/tag/v2.1.1"]'
    )

    anchor = soup.select_one(
        '.card-title a[href="https://github.com/0xShug0/audio.cpp"]'
    )
    assert anchor
    card = anchor.find_parent(class_='card')
    assert card
    links = {link.get('href') for link in card.select('a[href]')}
    assert {
        'https://github.com/0xShug0/audio.cpp/pull/155',
        'https://github.com/0xShug0/audio.cpp/blob/1778b23a5f6a4951c788e4bb0e7baa04f20012a2/docs/models/fun_asr_nano.md',
    } <= links
    card_text = card.get_text(' ', strip=True)
    for marker in ('Fun-ASR-Nano', 'CPU', 'CUDA', 'CLI', 'OpenAI'):
        assert marker in card_text


@pytest.mark.parametrize('relative', ('ecosystem.html', 'en/ecosystem.html'))
def test_gpt_sovits_card_exposes_the_merged_qwen3_runtime_contract(
    built_site, relative
):
    soup = read_soup(built_site / relative)
    anchor = soup.select_one(
        '.card-title a[href="https://github.com/RVC-Boss/GPT-SoVITS"]'
    )
    assert anchor
    card = anchor.find_parent(class_='card')
    assert card
    links = {link.get('href') for link in card.select('a[href]')}
    assert {
        'https://github.com/RVC-Boss/GPT-SoVITS/pull/2801',
        'https://github.com/RVC-Boss/GPT-SoVITS/pull/2803',
        'https://github.com/RVC-Boss/GPT-SoVITS/pull/2824',
    } <= links
    text = card.get_text(' ', strip=True)
    for marker in ('Fun-ASR-Nano', 'Transformers', '>=4.51,<5', 'Qwen3', 'KeyError'):
        assert marker in text


@pytest.mark.parametrize(
    ('relative', 'peer', 'markers'),
    (
        (
            'blog/subtitle-edit-fun-asr-sensevoice-local-subtitles.html',
            '/en/blog/subtitle-edit-fun-asr-sensevoice-local-subtitles.html',
            ('视频', '字幕', '当前是 beta 版本'),
        ),
        (
            'en/blog/subtitle-edit-fun-asr-sensevoice-local-subtitles.html',
            '/blog/subtitle-edit-fun-asr-sensevoice-local-subtitles.html',
            ('video', 'subtitles', 'currently a beta release'),
        ),
    ),
)
def test_subtitle_edit_blog_is_bilingual_and_evidence_backed(
    built_site, relative, peer, markers
):
    page = built_site / relative
    soup = read_soup(page)
    text = soup.get_text(' ', strip=True)

    assert soup.select_one('link[rel="canonical"]')['href'].endswith('/' + relative)
    assert soup.select_one(f'link[rel="alternate"][href$="{peer}"]')
    image = soup.select_one('article img[src]')
    assert image
    assert (built_site / image['src'].lstrip('/')).is_file()
    links = {link.get('href') for link in soup.select('a[href]')}
    assert {
        'https://github.com/SubtitleEdit/subtitleedit',
        'https://github.com/SubtitleEdit/subtitleedit/pull/13063',
        'https://github.com/SubtitleEdit/subtitleedit/releases/tag/v5.2.0-beta2',
        '/go/fun-asr',
        '/go/sensevoice',
    } <= links
    for marker in (
        'Video',
        'Audio to text',
        'Crisp ASR Fun-ASR Nano',
        'Crisp ASR SenseVoice',
        'Q4',
        'Q8',
        'F16',
        'Windows',
        'macOS',
        'Linux',
        'v5.2.0-beta2',
        *markers,
    ):
        assert marker in text


@pytest.mark.parametrize(
    ('relative', 'href'),
    (
        (
            'blog/index.html',
            '/blog/subtitle-edit-fun-asr-sensevoice-local-subtitles.html',
        ),
        (
            'en/blog/index.html',
            '/en/blog/subtitle-edit-fun-asr-sensevoice-local-subtitles.html',
        ),
    ),
)
def test_blog_indexes_surface_subtitle_edit_release(built_site, relative, href):
    soup = read_soup(built_site / relative)

    assert soup.select_one(f'a[href="{href}"]')


def test_complete_build_passes_output_validation(built_site):
    assert validate_output(built_site) == []


def test_broken_internal_link_fails_validation(built_site):
    page = built_site / 'index.html'
    html = page.read_text(encoding='utf-8')
    page.write_text(html.replace('/deploy/', '/missing/', 1), encoding='utf-8')

    assert '/index.html: broken internal link /missing/' in validate_output(built_site)


def test_missing_language_peer_fails_validation(built_site):
    (built_site / 'en/deploy/vllm.html').unlink()

    assert any('missing hreflang peer' in error for error in validate_output(built_site))


def test_duplicate_id_and_invalid_json_ld_fail_validation(built_site):
    page = built_site / 'deploy/vllm.html'
    html = page.read_text(encoding='utf-8')
    html = html.replace('id="commands"', 'id="main"', 1)
    html = html.replace('"@context": "https://schema.org"', 'not-json', 1)
    page.write_text(html, encoding='utf-8')
    errors = validate_output(built_site)

    assert '/deploy/vllm.html: duplicate id main' in errors
    assert '/deploy/vllm.html: invalid JSON-LD' in errors


def test_hashed_asset_tampering_fails_validation(built_site):
    manifest = __import__('json').loads(
        (built_site / 'deployment-manifest.json').read_text(encoding='utf-8')
    )
    asset = next(iter(manifest['assets']))
    with (built_site / asset.lstrip('/')).open('ab') as stream:
        stream.write(b'changed')

    assert any(f'asset hash mismatch {asset}' in error for error in validate_output(built_site))


@pytest.mark.parametrize(
    ('relative', 'peer', 'markers'),
    (
        (
            'blog/funasr-v1-4-3-pypi-release.html',
            '/en/blog/funasr-v1-4-3-pypi-release.html',
            ('FunASR v1.4.3', 'Silero VAD', '固定 K', '167', 'SHA256SUMS-v1.4.3'),
        ),
        (
            'en/blog/funasr-v1-4-3-pypi-release.html',
            '/blog/funasr-v1-4-3-pypi-release.html',
            ('FunASR v1.4.3', 'Silero VAD', 'fixed-K', '167', 'SHA256SUMS-v1.4.3'),
        ),
    ),
)
def test_v1_4_3_release_blog_is_bilingual_and_verifiable(
    built_site, relative, peer, markers
):
    soup = read_soup(built_site / relative)
    text = soup.get_text(' ', strip=True)

    assert soup.select_one('link[rel="canonical"]')['href'].endswith('/' + relative)
    assert soup.select_one(f'link[rel="alternate"][href$="{peer}"]')
    assert soup.select_one('script[type="application/ld+json"]')
    assert soup.select_one('a[href="https://github.com/modelscope/FunASR/releases/tag/v1.4.3"]')
    for marker in markers:
        assert marker in text

    root = ET.parse(built_site / 'sitemap.xml').getroot()
    namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    urls = {
        item.findtext('sitemap:loc', namespaces=namespace)
        for item in root.findall('sitemap:url', namespace)
    }
    assert f'https://www.funasr.com/{relative}' in urls


@pytest.mark.parametrize(
    ('relative', 'feature_href', 'history_href'),
    (
        (
            'blog/index.html',
            '/blog/funclip-v2-2-0-moss-speaker-clipping.html',
            '/blog/funasr-v1-4-5-pypi-llama-cpp-release.html',
        ),
        (
            'en/blog/index.html',
            '/en/blog/funclip-v2-2-0-moss-speaker-clipping.html',
            '/en/blog/funasr-v1-4-5-pypi-llama-cpp-release.html',
        ),
    ),
)
def test_blog_index_features_latest_release_and_preserves_history(
    built_site, relative, feature_href, history_href
):
    soup = read_soup(built_site / relative)
    feature = soup.select_one(f'.launch-feature a[href="{feature_href}"]')

    assert feature
    assert 'FunClip v2.2.0' in feature.get_text(' ', strip=True)
    history = soup.select_one('.previous-release')
    assert history
    assert history.select_one(f'a[href="{history_href}"]')
    history_text = history.get_text(' ', strip=True)
    assert 'v1.4.5' in history_text
    assert 'v1.4.3' in history_text
    assert 'v1.4.0' in history_text


@pytest.mark.parametrize(
    ('relative', 'peer', 'guide'),
    (
        (
            'blog/funclip-v2-2-0-moss-speaker-clipping.html',
            '/en/blog/funclip-v2-2-0-moss-speaker-clipping.html',
            '/deploy/moss-transcribe-diarize.html',
        ),
        (
            'en/blog/funclip-v2-2-0-moss-speaker-clipping.html',
            '/blog/funclip-v2-2-0-moss-speaker-clipping.html',
            '/en/deploy/moss-transcribe-diarize.html',
        ),
    ),
)
def test_funclip_v220_blog_builds_with_real_media_and_product_routes(
    built_site, relative, peer, guide
):
    soup = read_soup(built_site / relative)
    assert soup.select_one(f'link[rel="alternate"][href$="{peer}"]')
    image = soup.select_one('article img[src]')
    assert image
    assert (built_site / image['src'].lstrip('/')).is_file()
    assert soup.select_one(f'a[href="{guide}"]')
    assert soup.select_one(
        'a[href="https://github.com/modelscope/FunClip/releases/tag/v2.2.0"]'
    )
