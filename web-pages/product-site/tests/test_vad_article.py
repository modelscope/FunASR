"""Editorial invariants and illustrative arithmetic, not model inference."""
import json
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
SLUG = 'voice-activity-detection-python'
REF = '3747f8496815ee0932e1a227b17d57c27442d656'


@pytest.fixture(params=['', 'en/'], ids=['zh', 'en'])
def page(request):
    return request.param, BeautifulSoup((SITE/'legacy'/request.param/'blog'/(SLUG+'.html')).read_text(), 'html.parser')


def test_dates_and_identity(page):
    prefix, soup = page
    data = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert data['datePublished'] == '2026-06-22'
    assert data['dateModified'] == '2026-11-20'
    assert data['mainEntityOfPage']['@id'] == f'https://www.funasr.com/{prefix}blog/{SLUG}.html'
    assert soup.select_one('article h1').get_text() == data['headline']


def test_reader_outcome_and_honest_scope(page):
    _, soup = page
    for key in ['opening', 'outcome', 'scope', 'next-step']:
        assert len(soup.select_one(f'[data-editorial="{key}"]').get_text()) > 30
    assert soup.select_one('[data-example="illustrative"]')
    assert not soup.select('article pre')
    text = soup.select_one('article').get_text()
    for bad in ['110×', '0.12', 'CPU 也能实时跑', 'Everything below is real', 'zero extra dependencies', '3 行代码做 VAD']:
        assert bad not in text


def test_intervals_and_arithmetic(page):
    _, soup = page
    cells = [[x.get_text(strip=True) for x in row.select('td')] for row in soup.select('[data-example="intervals"] tbody tr')]
    assert len(cells) == 2
    intervals = [json.loads(row[0]) for row in cells]
    assert intervals == [[1000, 3000], [6000, 9000]]
    durations = [(end-start)/1000 for start, end in intervals]
    assert durations == [2, 3] and sum(durations) == 5
    assert all(str(duration) in row[2] for duration, row in zip(durations, cells))


def test_timeline_mapping_and_streaming_separation(page):
    _, soup = page
    assert soup.select_one('#keep-timeline')
    assert soup.select_one('[data-example="offset"]')
    assert soup.select_one('#streaming-boundary')
    text = soup.select_one('article').get_text()
    assert '[6.4, 6.9]' in text and '[start, -1]' in text and '[-1, end]' in text
    assert [round(6 + t, 1) for t in [.4, .9]] == [6.4, 6.9]


def test_agent_bounds_and_official_sources(page):
    _, soup = page
    assert soup.select_one('#agent-contract')
    assert '0 ≤ start < end' in soup.select_one('article').get_text()
    assert soup.select_one('#check-understanding')
    external = [urlsplit(x['href']) for x in soup.select('article a[href^="https:"]')]
    assert len(external) == 4
    assert all(x.netloc == 'github.com' and f'/blob/{REF}/' in x.path for x in external)


def test_metadata_and_manifest(page):
    prefix, soup = page
    data = json.loads((SITE/'data/blog.json').read_text())
    entry = next(x for x in data['articles'] if x['slug'] == SLUG)
    assert entry['reviewed'] is True and entry['category'] == 'explanations'
    assert entry['en' if prefix else 'zh']['title'] == soup.select_one('article h1').get_text()
    import hashlib
    key = prefix+'blog/'+SLUG+'.html'
    assert json.loads((SITE/'content/legacy-manifest.json').read_text())['files'][key] == hashlib.sha256((SITE/'legacy'/key).read_bytes()).hexdigest()
