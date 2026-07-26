from __future__ import annotations

import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup


SITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE_ROOT))

from build import build  # noqa: E402


def read_soup(path: Path) -> BeautifulSoup:
    return BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')


def test_build_emits_bilingual_product_routes(tmp_path):
    manifest = build(tmp_path)
    routes = {page['route'] for page in manifest['pages']}

    assert {'/', '/en/', '/deploy/', '/en/deploy/', '/404.html'} <= routes
    assert (tmp_path / 'index.html').is_file()
    assert (tmp_path / 'en/index.html').is_file()
    assert (tmp_path / 'deploy/index.html').is_file()
    assert (tmp_path / 'en/deploy/index.html').is_file()


def test_home_has_literal_brand_heading_and_next_section(tmp_path):
    build(tmp_path)
    soup = read_soup(tmp_path / 'index.html')

    assert soup.h1.get_text(strip=True) == 'FunASR'
    assert soup.select_one('[data-section="deployment-selector"]')
    assert soup.select_one('[data-section="deployment-matrix"]')
    assert soup.select_one('[data-section="api-contract"]')


def test_navigation_order_and_donors_last(tmp_path):
    build(tmp_path)

    for relative, expected_last in (
        ('index.html', '功德榜'),
        ('en/index.html', 'Donors'),
    ):
        soup = read_soup(tmp_path / relative)
        labels = [item.get_text(strip=True) for item in soup.select('[data-primary-nav] a')]
        assert labels[-1] == expected_last
        assert len(labels) == 8


def test_assets_are_local_and_content_hashed(tmp_path):
    build(tmp_path)
    soup = read_soup(tmp_path / 'index.html')
    asset_urls = [
        node.get('href') or node.get('src')
        for node in soup.select('link[rel="stylesheet"], script[src], img[src]')
    ]

    assert asset_urls
    assert all(url.startswith('/assets/') for url in asset_urls)
    assert all(re.search(r'\.[0-9a-f]{12}\.', url) for url in asset_urls)


def test_language_metadata_is_symmetric(tmp_path):
    build(tmp_path)
    zh = read_soup(tmp_path / 'index.html')
    en = read_soup(tmp_path / 'en/index.html')

    assert zh.select_one('link[rel="canonical"]')['href'] == 'https://www.funasr.com/'
    assert en.select_one('link[rel="canonical"]')['href'] == 'https://www.funasr.com/en/'
    assert zh.select_one('link[hreflang="en"]')['href'] == 'https://www.funasr.com/en/'
    assert en.select_one('link[hreflang="zh-CN"]')['href'] == 'https://www.funasr.com/'


def test_build_manifest_records_asset_hashes(tmp_path):
    manifest = build(tmp_path)

    assert manifest['schema_version'] == 1
    assert len(manifest['assets']) >= 3
    assert all(len(digest) == 64 for digest in manifest['assets'].values())
    assert (tmp_path / 'deployment-manifest.json').is_file()
