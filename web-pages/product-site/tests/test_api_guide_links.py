from pathlib import Path
import sys

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))


def test_api_guides_preserve_entries_and_localize_destinations(tmp_path):
    from link_api_guides import link_api_guides

    output, previous = tmp_path / 'output', tmp_path / 'previous'
    output.mkdir()
    (previous / 'zh').mkdir(parents=True)
    original = '<html><body><main><div id="api-welcome"><h2>API</h2></div><div class="api-detail" id="e3"><h2>AutoModel</h2><pre>a &lt; b</pre></div><a href="#e3">AutoModel</a></main><script>const x = "<unchanged>";</script></body></html>'
    (output / 'api.html').write_text(original)
    (previous / 'zh/api.html').write_text(original)
    link_api_guides(output, previous)
    first = {}
    for route, prefix, label in [('api.html', 'en/', 'Python SDK guide'), ('zh/api.html', '', 'Python SDK 指南')]:
        text = (output / route).read_text()
        first[route] = text
        soup = BeautifulSoup(text, 'html.parser')
        before = BeautifulSoup(original, 'html.parser')
        assert str(soup.select_one('#e3')) == str(before.select_one('#e3'))
        assert soup.script.string == before.script.string
        assert soup.select_one('a[href="#e3"]')
        links = soup.select('#api-practical-guides a')
        assert [a['href'] for a in links] == [
            f'https://www.funasr.com/{prefix}docs/native-transformers.html',
            f'https://www.funasr.com/{prefix}docs/python-api.html',
        ]
        assert links[1].get_text() == label
    link_api_guides(output, previous)
    assert all((output / route).read_text() == text for route, text in first.items())
    assert (previous / 'zh/api.html').read_text() == original


def test_invalid_legacy_page_does_not_partially_write(tmp_path):
    from link_api_guides import link_api_guides

    output, previous = tmp_path / 'output', tmp_path / 'previous'
    output.mkdir()
    (previous / 'zh').mkdir(parents=True)
    original = '<div id="api-welcome">API</div>'
    (output / 'api.html').write_text(original)
    (previous / 'zh/api.html').write_text('<div>No API welcome</div>')
    with pytest.raises(ValueError, match='api-welcome'):
        link_api_guides(output, previous)
    assert (output / 'api.html').read_text() == original
    assert not (output / 'zh/api.html').exists()


def test_pages_workflow_restores_only_the_legacy_api_input():
    workflow = (SITE.parents[1] / '.github/workflows/update-api-docs.yml').read_text()
    assert 'ref: gh-pages' in workflow
    assert 'sparse-checkout: zh/api.html' in workflow
    assert 'link_api_guides.py --output gh-pages-output --previous previous-pages' in workflow


@pytest.mark.parametrize('bad', [None, '<div id="api-welcome"></div><div id="api-welcome"></div>'])
def test_missing_or_duplicate_legacy_welcome_fails_closed(tmp_path, bad):
    from link_api_guides import link_api_guides

    output, previous = tmp_path / 'output', tmp_path / 'previous'
    output.mkdir()
    (previous / 'zh').mkdir(parents=True)
    original = '<div id="api-welcome">API</div>'
    (output / 'api.html').write_text(original)
    if bad is not None:
        (previous / 'zh/api.html').write_text(bad)
    with pytest.raises((ValueError, FileNotFoundError)):
        link_api_guides(output, previous)
    assert (output / 'api.html').read_text() == original
    assert not (output / 'zh/api.html').exists()
