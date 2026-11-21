"""Guard the bilingual CPU guide's scope, examples, sources and discovery."""
import hashlib
import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'lightweight-speech-recognition-cpu'
REF = '581150d8e1de2bc055cc390b6f2efc9ea78b48b7'


@pytest.fixture(params=['zh', 'en'])
def article(request):
    lang = request.param
    rel = ('en/' if lang == 'en' else '') + 'blog/' + SLUG + '.html'
    path = ROOT / 'legacy' / rel
    return lang, rel, path, BeautifulSoup(path.read_text(), 'html.parser')


def test_cpu_identity_and_revision(article):
    lang, rel, _, soup = article
    data = json.loads(soup.select_one('script[type="application/ld+json"]').string)
    assert data['datePublished'] == '2026-06-22'
    assert data['dateModified'] == '2026-11-22'
    assert soup.select_one('link[rel="canonical"]')['href'] == 'https://www.funasr.com/' + rel
    assert 'CPU' in soup.h1.get_text()
    assert soup.select_one('time[datetime="2026-06-22"]')
    assert soup.select_one('time[datetime="2026-11-22"]')


def test_cpu_reader_question_and_honest_scope(article):
    lang, _, _, soup = article
    body = soup.article.get_text(' ', strip=True)
    for anchor in ['choose-route', 'memory', 'read-timing', 'evaluate', 'agent-contract', 'sources']:
        assert soup.select_one('#' + anchor), anchor
    for term in ['FunASR', 'Fun-ASR', 'SenseVoice', 'FunClip', 'GGUF', 'RTF']:
        assert term in body
    assert not soup.article.select('pre'), 'Conceptual guide must not masquerade as a runnable tutorial'
    assert ('未新增模型推理' if lang == 'zh' else 'No new model inference') in body
    for claim in ['0.16s', '0.16 秒', '三倍', '3× more accurate', 'virtually unchanged', '准确率几乎不变']:
        assert claim not in body


def test_cpu_illustrative_timing_math(article):
    lang, _, _, soup = article
    table = soup.select_one('[data-example="timing"]')
    assert table
    assert ('虚构' if lang == 'zh' else 'invented') in soup.select_one('#read-timing').parent.get_text()
    rows = table.select('tbody tr')
    assert len(rows) == 3
    for row in rows:
        duration = float(row['data-audio-seconds'])
        processing = float(row['data-processing-seconds'])
        assert duration > 0 and processing >= 0
        assert float(row['data-rtf']) == pytest.approx(processing / duration)
    assert [(float(r['data-audio-seconds']), float(r['data-processing-seconds'])) for r in rows] == [(600, 120), (600, 360), (10, 2)]


def test_cpu_contract_includes_units_failures_and_authorization(article):
    lang, _, _, soup = article
    section = soup.select_one('#agent-contract').parent.get_text(' ', strip=True)
    for field in ['audio_duration_s', 'processing_s', 'request_latency_s', 'peak_rss_bytes', 'model_revision', 'runtime_version', 'threads', 'concurrency', 'warm_state', 'status', 'error']:
        assert field in section
    assert ('不得自动重试' if lang == 'zh' else 'Do not automatically retry') in section
    assert ('授权' if lang == 'zh' else 'authorization') in section
    assert ('秒' if lang == 'zh' else 'seconds') in section


def test_cpu_pinned_sources_and_next_step(article):
    _, _, _, soup = article
    urls = [a['href'] for a in soup.select('#sources ~ ul a[href]')]
    for path in ['runtime/llama.cpp/README.md', 'docs/model_selection.md', 'funasr/auto/auto_model.py']:
        assert any('/blob/' + REF + '/' + path in u for u in urls)
    assert any('speech-to-text-python-transcribe-audio.html' in a.get('href', '') for a in soup.article.select('a'))


def test_cpu_catalog_and_manifest_match(article):
    lang, rel, path, soup = article
    data = json.loads((ROOT / 'data/blog.json').read_text())
    entry = next(x for x in data['articles'] if x['slug'] == SLUG)
    assert entry['reviewed'] is True and entry['category'] == 'explanations'
    assert entry[lang]['summary'] == soup.select_one('meta[name="description"]')['content']
    assert entry[lang]['title'] == soup.h1.get_text(' ', strip=True)
    manifest = json.loads((ROOT / 'content/legacy-manifest.json').read_text())
    assert manifest['files'][rel] == hashlib.sha256(path.read_bytes()).hexdigest()
