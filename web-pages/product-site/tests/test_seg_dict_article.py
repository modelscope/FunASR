"""Execute each published dictionary inspector against real UTF-8 files."""
import contextlib
import io
import json
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
from build import build

SLUG = 'dictionary-input-checklist'
SOURCE = 'https://github.com/modelscope/FunASR/blob/d8a48a8af181dddf9b9f30aa50801c0ed6a6ffce/funasr/tokenizer/char_tokenizer.py'
FIELDS = ('blank_lines', 'empty_values', 'duplicate_keys', 'bom_lines', 'uppercase_keys')

def expected(entries=0, **changes):
    return dict(entries=entries, **{name: changes.get(name, []) for name in FIELDS})

def test_catalogue_discovers_dictionary_article_once():
    data = json.loads((SITE/'data/blog.json').read_text())
    rows = [x for x in data['articles'] if x['slug'] == SLUG]
    assert len(rows) == 1
    assert rows[0]['reviewed'] and rows[0]['category'] == 'explanations'
    assert SLUG not in data['selected']

@pytest.fixture(scope='module', params=['source', 'built'])
def tree(request, tmp_path_factory):
    if request.param == 'source':
        return SITE/'legacy'
    dest = tmp_path_factory.mktemp('dictionary-article')
    build(dest)
    return dest

@pytest.fixture(params=['', 'en/'])
def article(request, tree):
    prefix = request.param
    return prefix, BeautifulSoup((tree/prefix/'blog'/(SLUG+'.html')).read_text(), 'html.parser')

@pytest.fixture
def inspector(article):
    namespace = {'__name__': 'article_example'}
    code = article[1].select_one('code[data-language="python"]').get_text()
    exec(compile(code, 'article-dictionary-inspector', 'exec'), namespace)
    return namespace['inspect_seg_dict']

def test_metadata_and_source_links(article):
    prefix, soup = article
    meta = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert meta['datePublished'] == meta['dateModified'] == '2026-11-16'
    assert meta['headline'] == soup.select_one('article h1').get_text()
    assert soup.select_one('link[rel="canonical"]')['href'] == f'https://www.funasr.com/{prefix}blog/{SLUG}.html'
    links = {a['href'] for a in soup.select('article a[href]')}
    assert SOURCE in links and 'https://github.com/modelscope/FunASR/pull/3905' in links
    assert 'https://github.com/modelscope/FunASR/releases/tag/v1.4.26' in links
    assert soup.select_one('[data-editorial="boundary"]')
    assert len(soup.select('[data-editorial="next-step"] a')) == 1

@pytest.mark.parametrize('text,count', [
    ('hello h e l l o\nworld w o r l d\n', 2),
    ('\u4f60\u597d\t\u4f60\t\u597d\n', 1),
    ('# this is a dictionary entry\n', 1),
])
def test_clean_dictionary_and_no_implicit_comments(inspector, tmp_path, text, count):
    p = tmp_path/'dictionary.txt'; p.write_text(text, encoding='utf-8')
    before = p.read_bytes()
    assert inspector(p) == expected(count)
    assert p.read_bytes() == before

@pytest.mark.parametrize('text,want', [
    ('hello h\n\n  \t \nworld w\n', expected(2, blank_lines=[2, 3])),
    ('hello h\r\n \t\r\nworld w\r\n', expected(2, blank_lines=[2])),
    ('hello h\nworld w\nhello hh\nhello hhh\n', expected(2, duplicate_keys=[
        dict(key='hello', previous_line=1, line=3), dict(key='hello', previous_line=3, line=4)])),
    ('hello\n', expected(1, empty_values=[1])),
    ('\ufeffhello h\n', expected(1, bom_lines=[1])),
    ('HELLO h\nHello h\n', expected(2, uppercase_keys=[1, 2])),
])
def test_reports_exact_line_numbers_without_rewriting(inspector, tmp_path, text, want):
    p = tmp_path/'dictionary.txt'; p.write_bytes(text.encode('utf-8'))
    before = p.read_bytes()
    assert inspector(p) == want
    assert p.read_bytes() == before

def test_empty_dictionary_is_not_an_invented_entry(inspector, tmp_path):
    p = tmp_path/'empty.txt'; p.write_bytes(b'')
    assert inspector(p) == expected()
    assert p.read_bytes() == b''

def test_invalid_utf8_is_not_silently_replaced(inspector, tmp_path):
    p = tmp_path/'invalid.txt'; p.write_bytes(b'hello \xff\n')
    with pytest.raises(UnicodeDecodeError):
        inspector(p)
    assert p.read_bytes() == b'hello \xff\n'

def test_cli_runs_actual_published_code(article, tmp_path, monkeypatch):
    p = tmp_path/'input.txt'; p.write_text('hello h e l l o\n', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['inspect_dictionary.py', str(p)])
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(compile(article[1].select_one('code[data-language="python"]').get_text(), 'article-cli', 'exec'), {'__name__': '__main__'})
    assert json.loads(out.getvalue()) == expected(1)

@pytest.mark.parametrize('prefix', ['', 'en/'])
def test_archive_and_category_link_to_article(prefix, tmp_path):
    build(tmp_path)
    for category in ['archive', 'explanations']:
        soup = BeautifulSoup((tmp_path/prefix/'blog'/category/'index.html').read_text(), 'html.parser')
        cards = soup.select(f'a[href="/{prefix}blog/{SLUG}.html"]')
        assert len(cards) == 1 and cards[0].select_one('time[datetime="2026-11-16"]')
