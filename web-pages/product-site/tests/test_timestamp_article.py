"""Keep the bilingual timestamp explanation honest, linked and discoverable."""
import hashlib
import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SLUG = 'speech-to-text-timestamps-python'
REF = '68aa96ddb96b6ec9b2f50923b9c55d223ef483af'


@pytest.fixture(params=['zh', 'en'])
def article(request):
    lang = request.param
    rel = ('en/' if lang == 'en' else '') + 'blog/' + SLUG + '.html'
    path = ROOT / 'legacy' / rel
    return lang, rel, path, BeautifulSoup(path.read_text(), 'html.parser')


def test_timestamp_identity_and_revision(article):
    _, rel, _, soup = article
    data = json.loads(soup.select_one('script[type="application/ld+json"]').string)
    assert data['datePublished'] == '2026-06-23'
    assert data['dateModified'] == '2026-11-23'
    assert soup.select_one('link[rel="canonical"]')['href'] == 'https://www.funasr.com/' + rel
    assert soup.select_one('time[datetime="2026-06-23"]')
    assert soup.select_one('time[datetime="2026-11-23"]')


def test_timestamp_reader_navigation_resolves(article):
    _, _, _, soup = article
    links = soup.article.select('a[href^="#"]')
    assert len(links) >= 4
    for link in links:
        anchor = link['href'][1:]
        assert anchor and len(soup.article.find_all(id=anchor)) == 1


def test_timestamp_illustrative_timeline_math(article):
    lang, _, _, soup = article
    section = soup.select_one('#read-timeline').parent
    rows = section.select('[data-example="timeline"] tbody tr')
    assert len(rows) == 3
    expected = [(0, 1200, 1500), (30000, 1200, 1500), (60000, 250, 750)]
    for row, (offset, start, end) in zip(rows, expected):
        assert int(row['data-offset-ms']) == offset
        assert int(row['data-start-ms']) == start
        assert int(row['data-end-ms']) == end
        assert int(row['data-parent-start-ms']) == offset + start
        assert int(row['data-parent-end-ms']) == offset + end
        assert 0 <= start < end
        assert float(row['data-parent-start-s']) == pytest.approx((offset + start) / 1000)


def test_timestamp_manifest_matches_actual_bytes(article):
    _, rel, path, _ = article
    manifest = json.loads((ROOT / 'content/legacy-manifest.json').read_text())
    assert manifest['files'][rel] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_timestamp_pinned_sources_and_next_steps(article):
    _, _, _, soup = article
    urls = [a['href'] for a in soup.select('#sources ~ ul a[href]')]
    for path in ['funasr/models/paraformer/model.py', 'funasr/utils/timestamp_tools.py', 'funasr/utils/postprocess_utils.py', 'funasr/auto/auto_model.py']:
        assert any('/blob/' + REF + '/' + path in u for u in urls)
    links = [a.get('href', '') for a in soup.article.select('a')]
    assert any('speech-to-text-python-transcribe-audio.html' in u for u in links)
    assert any('generate-subtitles-srt-vtt-from-audio-video.html' in u for u in links)


def test_timestamp_catalog_matches_rendered_metadata(article):
    lang, rel, path, soup = article
    data = json.loads((ROOT / 'data/blog.json').read_text())
    entry = next(x for x in data['articles'] if x['slug'] == SLUG)
    assert entry['reviewed'] is True and entry['category'] == 'explanations'
    assert entry[lang]['title'] == soup.h1.get_text(' ', strip=True)
    assert entry[lang]['summary'] == soup.select_one('meta[name="description"]')['content']
