"""Execute the published WAV inspector on real, hand-checked binary fixtures."""
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import wave

from bs4 import BeautifulSoup
import pytest

SITE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SITE))
from build import build

SLUG='wav-pcm-input-checklist'
SOURCE='https://github.com/modelscope/FunASR/blob/173cbee304836dae94079c8f600272f3637451a4/funasr/utils/load_utils.py'

def test_catalogue_discovers_input_article_once():
    data=json.loads((SITE/'data/blog.json').read_text())
    rows=[row for row in data['articles'] if row['slug']==SLUG]
    assert len(rows)==1
    assert rows[0]['reviewed'] and rows[0]['category']=='explanations'
    assert SLUG not in data['selected']

@pytest.fixture(scope='module',params=['source','built'])
def tree(request,tmp_path_factory):
    if request.param=='source':return SITE/'legacy'
    dest=tmp_path_factory.mktemp('wav-inspector-article')
    build(dest)
    return dest

@pytest.fixture(params=['','en/'])
def article(request,tree):
    prefix=request.param
    return prefix,BeautifulSoup((tree/prefix/'blog'/(SLUG+'.html')).read_text(),'html.parser')

@pytest.fixture
def inspector(article):
    _,soup=article
    namespace={'__name__':'article_example'}
    exec(compile(soup.select_one('code[data-language="python"]').get_text(),'article-wav-inspector','exec'),namespace)
    return namespace['inspect_pcm16_wav']

def write_wav(path,samples,channels=1,rate=16000,width=2):
    with wave.open(str(path),'wb') as out:
        out.setnchannels(channels);out.setsampwidth(width);out.setframerate(rate)
        raw=bytes(samples) if width==1 else struct.pack('<'+'h'*len(samples),*samples)
        out.writeframes(raw)

def test_metadata_and_source_links(article):
    prefix,soup=article
    meta=json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert meta['datePublished']==meta['dateModified']=='2026-11-15'
    assert meta['headline']==soup.select_one('article h1').get_text()
    assert soup.select_one('link[rel="canonical"]')['href']==f'https://www.funasr.com/{prefix}blog/{SLUG}.html'
    links={a['href'] for a in soup.select('article a[href]')}
    assert SOURCE in links and 'https://github.com/modelscope/FunASR/pull/3899' in links
    assert 'https://github.com/modelscope/FunASR/releases/tag/v1.4.26' in links
    assert soup.select_one('[data-editorial="boundary"]')
    assert len(soup.select('[data-editorial="next-step"] a'))==1

@pytest.mark.parametrize('samples,channels,rate,frames,duration,peak',[
    ([-32768,-16384,0,32767],1,8000,4,0.0005,1.0),
    ([32767,0,0,0],1,16000,4,0.00025,0.999969482421875),
    ([-16384,16384,0,8192],2,16000,2,0.000125,0.5),
    ([0,0,0,0],2,8000,2,0.00025,0.0),
])
def test_real_wav_reports_frames_not_scalar_samples(inspector,tmp_path,samples,channels,rate,frames,duration,peak):
    path=tmp_path/'audio.wav';write_wav(path,samples,channels,rate)
    before=path.read_bytes()
    assert inspector(path)==dict(channels=channels,sample_rate=rate,frames=frames,duration_seconds=duration,peak_abs=peak)
    assert path.read_bytes()==before

@pytest.mark.parametrize('problem',['empty','8bit','truncated-sample','truncated-frame','not-wav'])
def test_invalid_or_unsupported_input_is_not_accepted(inspector,tmp_path,problem):
    path=tmp_path/'bad.wav'
    if problem=='empty':write_wav(path,[])
    elif problem=='8bit':write_wav(path,[0,128,255],width=1)
    elif problem=='not-wav':path.write_bytes(b'not an audio container')
    else:
        write_wav(path,[1,2,3,4],channels=2)
        path.write_bytes(path.read_bytes()[:(-1 if problem=='truncated-sample' else -4)])
    before=path.read_bytes()
    with pytest.raises((ValueError,wave.Error,EOFError)):
        inspector(path)
    assert path.read_bytes()==before

def test_inspector_reads_past_first_chunk(inspector,tmp_path):
    path=tmp_path/'long.wav'
    write_wav(path,[0]*40000+[-32768],rate=16000)
    assert inspector(path)==dict(channels=1,sample_rate=16000,frames=40001,duration_seconds=2.5000625,peak_abs=1.0)

def test_cli_example_executes(article,tmp_path,monkeypatch):
    _,soup=article
    path=tmp_path/'input.wav';write_wav(path,[0,16384],rate=8000)
    monkeypatch.setattr(sys,'argv',['inspect_wav.py',str(path)])
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(compile(soup.select_one('code[data-language="python"]').get_text(),'article-wav-cli','exec'),{'__name__':'__main__'})
    assert json.loads(out.getvalue())==dict(channels=1,sample_rate=8000,frames=2,duration_seconds=0.00025,peak_abs=0.5)

@pytest.mark.parametrize('prefix',['','en/'])
def test_archive_and_category_link_to_article(prefix,tmp_path):
    build(tmp_path)
    for category in ['archive','explanations']:
        soup=BeautifulSoup((tmp_path/prefix/'blog'/category/'index.html').read_text(),'html.parser')
        cards=soup.select(f'a[href="/{prefix}blog/{SLUG}.html"]')
        assert len(cards)==1 and cards[0].select_one('time[datetime="2026-11-15"]')
