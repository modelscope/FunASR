"""The model-selection article must match source and run without loading models."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parents[1]
sys.path.insert(0, str(SITE))
from build import build

SLUG = 'websocket-offline-model-options'
SOURCE = 'https://github.com/modelscope/FunASR/blob/9fbcf4d89937318758290c9125d746b8962094db/runtime/python/websocket/funasr_wss_server.py'

def test_catalogue_contains_model_article_once():
    data = json.loads((SITE/'data/blog.json').read_text())
    rows = [x for x in data['articles'] if x['slug'] == SLUG]
    assert len(rows) == 1
    assert rows[0]['reviewed'] and rows[0]['category'] == 'explanations'
    assert SLUG not in data['selected']

@pytest.fixture(scope='module', params=['source', 'built'])
def tree(request, tmp_path_factory):
    if request.param == 'source':
        return SITE/'legacy'
    dest = tmp_path_factory.mktemp('model-options-article')
    build(dest)
    return dest

@pytest.fixture(params=['', 'en/'])
def article(request, tree):
    prefix = request.param
    return prefix, BeautifulSoup((tree/prefix/'blog'/(SLUG+'.html')).read_text(), 'html.parser')

def test_metadata_sources_and_scope(article):
    prefix, soup = article
    meta = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert meta['datePublished'] == meta['dateModified'] == '2026-11-14'
    assert meta['headline'] == soup.select_one('article h1').get_text()
    assert soup.select_one('link[rel="canonical"]')['href'] == f'https://www.funasr.com/{prefix}blog/{SLUG}.html'
    links = {a['href'] for a in soup.select('article a[href]')}
    assert SOURCE in links
    assert 'https://github.com/modelscope/FunASR/pull/3888' in links
    assert soup.select_one('[data-editorial="boundary"]')
    assert len(soup.select('[data-editorial="next-step"] a')) == 1
    text = soup.select_one('article').get_text(' ', strip=True)
    for term in ['--asr_model', '--asr_model_revision', '--asr_model_online', 'SHA256', 'v2.0.4']:
        assert term in text

def test_source_audit_example_executes_without_server_import(article, monkeypatch):
    _, soup = article
    code = soup.select_one('code[data-language="python"]').get_text()
    server = ROOT/'runtime/python/websocket/funasr_wss_server.py'
    monkeypatch.setattr(sys, 'argv', ['audit_model_options.py', str(server)])
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(compile(code, 'article-source-audit', 'exec'), {})
    result = json.loads(out.getvalue())
    assert result['sha256'] == hashlib.sha256(server.read_bytes()).hexdigest()
    assert result['offline_model'] == 'args.asr_model'
    assert result['offline_revision'] == 'args.asr_model_revision'

@pytest.mark.parametrize('prefix', ['', 'en/'])
def test_archive_and_explanations_discover_article(prefix, tmp_path):
    build(tmp_path)
    for category in ['archive', 'explanations']:
        soup = BeautifulSoup((tmp_path/prefix/'blog'/category/'index.html').read_text(), 'html.parser')
        cards = soup.select(f'a[href="/{prefix}blog/{SLUG}.html"]')
        assert len(cards) == 1
        assert cards[0].select_one('time[datetime="2026-11-14"]')
