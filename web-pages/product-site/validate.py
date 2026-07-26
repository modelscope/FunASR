"""Validate a complete FunASR product-site output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup


BASE_URL = 'https://www.funasr.com'
LOCAL_HOSTS = {'funasr.com', 'www.funasr.com'}
REQUIRED_DEPLOYMENT_SECTIONS = {
    'fit',
    'commands',
    'smoke-test',
    'security',
    'limitations',
    'operations',
    'evidence',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def page_route(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    if relative.name == 'index.html':
        parent = relative.parent.as_posix()
        return '/' if parent == '.' else f'/{parent}/'
    return f'/{relative.as_posix()}'


def route_path(root: Path, route: str) -> Path:
    clean = unquote(route.split('?', 1)[0].split('#', 1)[0])
    normalized = posixpath.normpath(clean)
    if clean.endswith('/'):
        normalized = normalized.rstrip('/') + '/index.html'
    elif normalized == '/':
        normalized = '/index.html'
    return root / normalized.lstrip('/')


def _local_route(reference: str, current_route: str) -> str | None:
    if not reference or reference.startswith(('#', 'mailto:', 'tel:', 'javascript:', 'data:')):
        return None
    resolved = urlparse(urljoin(f'{BASE_URL}{current_route}', reference))
    if resolved.scheme not in {'http', 'https'} or resolved.netloc not in LOCAL_HOSTS:
        return None
    route = resolved.path or '/'
    if resolved.query:
        route += f'?{resolved.query}'
    return route


def _html_pages(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob('*.html') if path.is_file())


def _canonical_route(soup: BeautifulSoup) -> str | None:
    node = soup.select_one('link[rel="canonical"][href]')
    if node is None:
        return None
    parsed = urlparse(str(node['href']))
    if parsed.netloc not in LOCAL_HOSTS:
        return None
    return parsed.path or '/'


def _sitemap_routes(root: Path) -> tuple[set[str], list[str]]:
    sitemap = root / 'sitemap.xml'
    if not sitemap.is_file():
        return set(), ['sitemap.xml: missing']
    try:
        tree = ET.parse(sitemap)
    except ET.ParseError:
        return set(), ['sitemap.xml: invalid XML']
    locations = []
    for node in tree.findall('.//{*}loc'):
        parsed = urlparse(node.text or '')
        if parsed.netloc in LOCAL_HOSTS:
            locations.append(parsed.path or '/')
    duplicates = sorted(route for route, count in Counter(locations).items() if count > 1)
    return set(locations), [f'sitemap.xml: duplicate route {route}' for route in duplicates]


def _validate_page(root: Path, path: Path) -> list[str]:
    errors: list[str] = []
    route = page_route(root, path)
    label = f'/{path.relative_to(root).as_posix()}'
    try:
        soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
    except UnicodeDecodeError:
        return [f'{label}: not UTF-8']

    ids = [str(node['id']) for node in soup.select('[id]')]
    for item, count in sorted(Counter(ids).items()):
        if count > 1:
            errors.append(f'{label}: duplicate id {item}')

    for node in soup.select('[href], [src]'):
        attribute = 'href' if node.has_attr('href') else 'src'
        reference = str(node[attribute])
        local_route = _local_route(reference, route)
        if local_route is None or local_route.startswith('/go/'):
            continue
        if not route_path(root, local_route).is_file():
            errors.append(f'{label}: broken internal link {local_route.split("?", 1)[0]}')

    if path.name != '404.html':
        for alternate in soup.select('link[rel="alternate"][hreflang][href]'):
            peer = _local_route(str(alternate['href']), route)
            if peer is not None and not route_path(root, peer).is_file():
                errors.append(f'{label}: missing hreflang peer {peer}')

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            json.loads(script.string or script.get_text())
        except (json.JSONDecodeError, TypeError):
            errors.append(f'{label}: invalid JSON-LD')

    is_product_page = soup.body is not None and soup.body.has_attr('data-route')
    if is_product_page:
        for image in soup.find_all('img'):
            if not image.has_attr('alt'):
                errors.append(f'{label}: image missing alt text')
        for script in soup.select('script[src]'):
            if _local_route(str(script['src']), route) is None:
                errors.append(f'{label}: external script dependency {script["src"]}')
        for stylesheet in soup.select('link[rel="stylesheet"][href]'):
            if _local_route(str(stylesheet['href']), route) is None:
                errors.append(f'{label}: external stylesheet dependency {stylesheet["href"]}')

    if (
        route.startswith(('/deploy/', '/en/deploy/'))
        and not route.endswith('/')
        and route.count('/') in {2, 3}
    ):
        sections = {
            str(node['data-section'])
            for node in soup.select('[data-section]')
        }
        for section in sorted(REQUIRED_DEPLOYMENT_SECTIONS - sections):
            errors.append(f'{label}: missing deployment section {section}')
        if not soup.select_one('[data-field="verified-date"]'):
            errors.append(f'{label}: missing verified date')
    return errors


def validate_output(output_dir: Path) -> list[str]:
    """Return deterministic validation errors for a built output directory."""
    root = Path(output_dir).resolve()
    errors: list[str] = []
    manifest_path = root / 'deployment-manifest.json'
    if not manifest_path.is_file():
        return ['deployment-manifest.json: missing']
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return ['deployment-manifest.json: invalid JSON']

    for page in _html_pages(root):
        errors.extend(_validate_page(root, page))

    manifest_routes = {str(page['route']) for page in manifest.get('pages', [])}
    for route in sorted(manifest_routes):
        if not route_path(root, route).is_file():
            errors.append(f'deployment-manifest.json: missing route {route}')

    for asset, expected in sorted(manifest.get('assets', {}).items()):
        path = root / asset.lstrip('/')
        if not path.is_file():
            errors.append(f'deployment-manifest.json: missing asset {asset}')
        elif sha256(path) != expected:
            errors.append(f'deployment-manifest.json: asset hash mismatch {asset}')

    sitemap_routes, sitemap_errors = _sitemap_routes(root)
    errors.extend(sitemap_errors)
    expected_sitemap = set()
    for page in _html_pages(root):
        if page.name == '404.html':
            continue
        soup = BeautifulSoup(page.read_text(encoding='utf-8'), 'html.parser')
        canonical = _canonical_route(soup)
        if canonical:
            expected_sitemap.add(canonical)
    if sitemap_routes != expected_sitemap:
        for route in sorted(expected_sitemap - sitemap_routes):
            errors.append(f'sitemap.xml: missing route {route}')
        for route in sorted(sitemap_routes - expected_sitemap):
            errors.append(f'sitemap.xml: unexpected route {route}')
    for route in sorted(sitemap_routes):
        if not route_path(root, route).is_file():
            errors.append(f'sitemap.xml: route does not resolve {route}')

    robots = root / 'robots.txt'
    if not robots.is_file():
        errors.append('robots.txt: missing')
    elif f'Sitemap: {BASE_URL}/sitemap.xml' not in robots.read_text(encoding='utf-8'):
        errors.append('robots.txt: missing canonical sitemap')

    return sorted(set(errors))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args(argv)
    errors = validate_output(args.output_dir)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    page_count = len(_html_pages(args.output_dir.resolve()))
    print(f'validated {page_count} pages')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
