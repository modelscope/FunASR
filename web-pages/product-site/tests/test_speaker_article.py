"""Editorial schema and teaching arithmetic, not pretrained inference."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
SLUG = 'funasr-speaker-diarization'
REF = 'aa3d789cf0c99ba955178513c8471e38d68efca9'


@pytest.fixture(params=['', 'en/'], ids=['zh', 'en'])
def page(request):
    return request.param, BeautifulSoup((SITE/'legacy'/request.param/'blog'/(SLUG+'.html')).read_text(), 'html.parser')


def test_dates_and_identity(page):
    prefix, soup = page
    data = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert data['datePublished'] == '2026-06-17'
    assert data['dateModified'] == '2026-11-21'
    assert data['mainEntityOfPage']['@id'] == f'https://www.funasr.com/{prefix}blog/{SLUG}.html'
    assert soup.select_one('article h1').get_text() == data['headline']


def test_reader_outcome_and_honest_scope(page):
    _, soup = page
    for key in ['opening', 'outcome', 'scope', 'next-step']:
        assert len(soup.select_one(f'[data-editorial="{key}"]').get_text()) > 30
    assert soup.select_one('[data-example="illustrative"]')
    assert not soup.select('article pre')
    text = soup.select_one('article').get_text()
    for bad in ['227', '77 sentences', '11 different speakers', '中文更准', 'Stronger on Chinese', 'Swap model=', '一个 Python 依赖']:
        assert bad not in text


def test_recording_local_example(page):
    _, soup = page
    rows = soup.select('[data-example="turns"] tbody tr')
    assert len(rows) == 3
    cells = [[x.get_text(strip=True) for x in row.select('td')] for row in rows]
    assert [json.loads(x[0]) for x in cells] == [[0, 3000], [3000, 7000], [7000, 9000]]
    assert [int(x[1]) for x in cells] == [0, 1, 0]
    assert soup.select_one('#identity-boundary')
    assert soup.select_one('[data-contract="recording-scoped-key"]')


def test_sentence_assignment_is_not_speaker_purity(page):
    _, soup = page
    span = soup.select_one('[data-example="overlap"]')
    assert json.loads(span['data-sentence-ms']) == [0, 8000]
    turns = json.loads(span['data-turns-seconds'])
    assert turns == [[0, 3, 0], [3, 8, 1]]
    totals = {speaker: end-start for start, end, speaker in turns}
    assert max(totals, key=totals.get) == int(span['data-selected-speaker']) == 1
    assert soup.select_one('#zero-is-not-confidence')


def test_agent_bounds_and_pinned_sources(page):
    _, soup = page
    for ident in ['agent-contract', 'review-workflow', 'model-boundaries', 'check-understanding']:
        assert soup.select_one('#'+ident)
    for key in ['speaker-id', 'time-units', 'missing-result', 'authorization', 'retry']:
        assert soup.select_one(f'[data-contract="{key}"]')
    external = [urlsplit(x['href']) for x in soup.select('article a[href^="https:"]')]
    assert len(external) >= 4
    assert all(x.netloc == 'github.com' and f'/blob/{REF}/' in x.path for x in external)


def test_metadata_and_manifest(page):
    prefix, soup = page
    data = json.loads((SITE/'data/blog.json').read_text())
    entry = next(x for x in data['articles'] if x['slug'] == SLUG)
    assert entry['reviewed'] is True and entry['category'] == 'explanations'
    assert entry['en' if prefix else 'zh']['title'] == soup.select_one('article h1').get_text()
    key = prefix+'blog/'+SLUG+'.html'
    assert json.loads((SITE/'content/legacy-manifest.json').read_text())['files'][key] == hashlib.sha256((SITE/'legacy'/key).read_bytes()).hexdigest()
