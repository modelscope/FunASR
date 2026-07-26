from __future__ import annotations

import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup


SITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE_ROOT))

from build import build  # noqa: E402
from registry import BENCHMARK_FIELDS, load_registry  # noqa: E402


@pytest.fixture
def built_site(tmp_path):
    build(tmp_path)
    return tmp_path


def read_soup(path: Path) -> BeautifulSoup:
    return BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')


def route_path(root: Path, route: str) -> Path:
    if route.endswith('/'):
        return root / route.lstrip('/') / 'index.html'
    return root / route.lstrip('/')


def test_every_deployment_page_has_operational_contract(built_site):
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')

    for entry in registry['deployments']:
        for language in ('zh', 'en'):
            page = route_path(built_site, entry['routes'][language])
            soup = read_soup(page)
            assert soup.select_one('[data-field="verified-date"]')
            assert soup.select_one('[data-section="fit"]')
            assert soup.select_one('[data-section="commands"]')
            assert soup.select_one('[data-section="smoke-test"]')
            assert soup.select_one('[data-section="security"]')
            assert soup.select_one('[data-section="limitations"]')
            assert soup.select_one('[data-section="operations"]')
            assert soup.select_one('[data-section="evidence"]')


def test_detail_commands_come_from_registry(built_site):
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')

    for entry in registry['deployments']:
        soup = read_soup(route_path(built_site, entry['routes']['en']))
        rendered = soup.get_text('\n')
        for command in entry['commands']['smoke']:
            assert command in rendered


def test_benchmark_rows_have_complete_conditions(built_site):
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')
    records = [record for entry in registry['deployments'] for record in entry['benchmarks']]
    soup = read_soup(built_site / 'en/benchmarks.html')
    rows = soup.select('[data-benchmark-record]')

    assert len(records) >= 3
    assert len(rows) == len(records)
    for record in records:
        assert all(record.get(field) for field in BENCHMARK_FIELDS)
    for row in rows:
        assert row.select_one('[data-field="benchmark-hardware"]')
        assert row.select_one('[data-field="benchmark-settings"]')
        assert row.select_one('[data-field="benchmark-timing"]')
        assert row.select_one('[data-field="benchmark-source"]')
        assert row.select_one('[data-field="benchmark-qualification"]')


def test_benchmark_page_warns_against_cross_profile_claims(built_site):
    for relative in ('benchmarks.html', 'en/benchmarks.html'):
        soup = read_soup(built_site / relative)
        warning = soup.select_one('[data-benchmark-warning]')
        assert warning
        assert 'RTFx' in warning.get_text()


def test_manifest_contains_all_detail_and_benchmark_routes(built_site):
    manifest = __import__('json').loads(
        (built_site / 'deployment-manifest.json').read_text(encoding='utf-8')
    )
    routes = {page['route'] for page in manifest['pages']}
    registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')
    expected = {
        entry['routes'][language]
        for entry in registry['deployments']
        for language in ('zh', 'en')
    }
    expected.update({'/benchmarks.html', '/en/benchmarks.html'})

    assert expected <= routes


def test_old_llama_routes_point_to_product_pages(built_site):
    for relative, expected in (
        ('llama-cpp.html', '/deploy/llama-cpp.html'),
        ('en/llama-cpp.html', '/en/deploy/llama-cpp.html'),
    ):
        soup = read_soup(built_site / relative)
        assert soup.select_one('link[rel="canonical"]')['href'].endswith(expected)
        assert soup.select_one('.nav-links a[href$="/deploy/"]') or soup.select_one(
            '.nav-links a[href$="/en/deploy/"]'
        )
