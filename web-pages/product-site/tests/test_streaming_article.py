"""Pin streaming-guide identity, protocol examples and discoverability."""
import hashlib
import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'funasr-realtime-streaming-asr'
REF = 'ee46df3b7bca7d956af6b3c5deff23599c1de2cc'


@pytest.fixture(params=['zh', 'en'])
def article(request):
    lang = request.param
    rel = ('en/' if lang == 'en' else '') + 'blog/' + SLUG + '.html'
    path = ROOT / 'legacy' / rel
    return lang, rel, path, BeautifulSoup(path.read_text(), 'html.parser')


def test_streaming_identity_and_revision(article):
    _, rel, _, soup = article
    data = json.loads(soup.select_one('script[type="application/ld+json"]').string)
    assert data['datePublished'] == '2026-06-17'
    assert data['dateModified'] == '2026-11-24'
    assert soup.select_one('link[rel="canonical"]')['href'] == 'https://www.funasr.com/' + rel
    assert soup.select_one('time[datetime="2026-06-17"]')
    assert soup.select_one('time[datetime="2026-11-24"]')


def test_streaming_navigation_and_concept_boundaries(article):
    _, _, _, soup = article
    for name in ['ecosystem', 'read-messages', 'latency', 'agent-contract', 'sources']:
        assert len(soup.article.find_all(id=name)) == 1
    links = soup.article.select('a[href^="#"]')
    assert len(links) >= 4
    for link in links:
        assert len(soup.article.find_all(id=link['href'][1:])) == 1
    assert not soup.article.select('pre')


def test_streaming_message_phases_are_not_cumulative_text(article):
    _, _, _, soup = article
    rows = soup.select('[data-example="messages"] tbody tr')
    assert [r['data-kind'] for r in rows] == ['partial', 'segment-final', 'partial', 'input-complete']
    assert [r['data-mode'] for r in rows] == ['2pass-online', '2pass-offline', '2pass-online', '2pass']
    assert [r['data-is-final'] for r in rows] == ['false', 'true', 'false', 'true']
    assert [r.get('data-is-end', 'false') for r in rows] == ['false', 'false', 'false', 'true']
    assert rows[-1]['data-has-text'] == 'false'


def test_streaming_manifest_matches_actual_bytes(article):
    _, rel, path, _ = article
    manifest = json.loads((ROOT / 'content/legacy-manifest.json').read_text())
    assert manifest['files'][rel] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_streaming_pinned_sources_and_next_steps(article):
    _, _, _, soup = article
    urls = [a['href'] for a in soup.select('#sources ~ ul a[href]')]
    for path in ['runtime/python/websocket/funasr_wss_server.py', 'runtime/python/websocket/funasr_wss_client.py', 'funasr/models/paraformer_streaming/model.py', 'runtime/python/websocket/README.md']:
        assert any('/blob/' + REF + '/' + path in u for u in urls)
    links = [a.get('href', '') for a in soup.article.select('a')]
    assert any('speech-to-text-python-transcribe-audio.html' in u for u in links)
    assert any('which-funasr-model.html' in u for u in links)


def test_streaming_catalog_matches_rendered_metadata(article):
    lang, _, _, soup = article
    data = json.loads((ROOT / 'data/blog.json').read_text())
    entry = next(x for x in data['articles'] if x['slug'] == SLUG)
    assert entry['reviewed'] is True and entry['category'] == 'explanations'
    assert entry[lang]['title'] == soup.h1.get_text(' ', strip=True)
    assert entry[lang]['summary'] == soup.select_one('meta[name="description"]')['content']
