"""Build the dependency-free FunASR product site."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from legacy import normalize_document
from registry import load_registry, validate_registry
from selector import MATCH_WEIGHTS


SITE_ROOT = Path(__file__).resolve().parent
BASE_URL = 'https://www.funasr.com'
GITHUB_REPOSITORY_URL = 'https://github.com/modelscope/FunASR'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def route_path(output_dir: Path, route: str) -> Path:
    if route == '/':
        return output_dir / 'index.html'
    if route.endswith('/'):
        return output_dir / route.lstrip('/') / 'index.html'
    return output_dir / route.lstrip('/')


def canonical_url(route: str) -> str:
    return f'{BASE_URL}{route}'


def legacy_route(relative: Path) -> str:
    """Convert a legacy HTML path to its public route."""
    route = f'/{relative.as_posix()}'
    if relative.name == 'index.html':
        route = route[:-len('index.html')]
    return route


def _write_sitemap(stage: Path, last_modified_by_route: dict[str, str]) -> None:
    routes: set[str] = set()
    for page in sorted(stage.rglob('*.html')):
        if page.name == '404.html':
            continue
        soup = BeautifulSoup(page.read_text(encoding='utf-8'), 'html.parser')
        canonical = soup.select_one('link[rel="canonical"][href]')
        if canonical is None:
            continue
        href = str(canonical['href'])
        if not href.startswith(BASE_URL):
            continue
        route = href[len(BASE_URL):] or '/'
        routes.add(route)

    ET.register_namespace('', 'http://www.sitemaps.org/schemas/sitemap/0.9')
    urlset = ET.Element('{http://www.sitemaps.org/schemas/sitemap/0.9}urlset')
    for route in sorted(routes):
        url = ET.SubElement(urlset, '{http://www.sitemaps.org/schemas/sitemap/0.9}url')
        location = ET.SubElement(url, '{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
        location.text = canonical_url(route)
        if last_modified := last_modified_by_route.get(route):
            modified = ET.SubElement(
                url, '{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod'
            )
            modified.text = last_modified
    ET.indent(urlset, space='  ')
    tree = ET.ElementTree(urlset)
    tree.write(stage / 'sitemap.xml', encoding='utf-8', xml_declaration=True)


def _copy_hashed_assets(stage: Path) -> tuple[dict[str, str], dict[str, str]]:
    urls: dict[str, str] = {}
    hashes: dict[str, str] = {}
    source_root = SITE_ROOT / 'assets'
    for source in sorted(path for path in source_root.rglob('*') if path.is_file()):
        relative = source.relative_to(source_root)
        digest = sha256(source)
        destination_name = f'{source.stem}.{digest[:12]}{source.suffix}'
        destination_relative = relative.with_name(destination_name)
        destination = stage / 'assets' / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        key = relative.as_posix()
        url = f'/assets/{destination_relative.as_posix()}'
        urls[key] = url
        hashes[url] = digest
    return urls, hashes


def _navigation(data: dict[str, Any], language: str) -> list[dict[str, str]]:
    suffix = 'zh' if language == 'zh' else 'en'
    return [
        {
            'id': item['id'],
            'label': item[language],
            'href': item[f'href_{suffix}'],
        }
        for item in data['items']
    ]


def _page_context(
    *,
    language: str,
    route: str,
    peer_route: str,
    title: str,
    description: str,
    date_modified: str,
    navigation: dict[str, Any],
    assets: dict[str, str],
) -> dict[str, Any]:
    return {
        'language': language,
        'html_language': 'zh-CN' if language == 'zh' else 'en',
        'route': route,
        'peer_route': peer_route,
        'canonical': canonical_url(route),
        'peer_canonical': canonical_url(peer_route),
        'title': title,
        'description': description,
        'date_modified': date_modified,
        'navigation': _navigation(navigation, language),
        'assets': assets,
        'github_repository_url': GITHUB_REPOSITORY_URL,
        'github_url': '/go/github',
        'docs_url': '/go/docs',
        'releases_url': '/go/releases',
    }


def _selector_payload(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'weights': MATCH_WEIGHTS,
        'entries': [
            {
                'id': entry['id'],
                'routes': entry['routes'],
                'rank': entry['selector_rank'],
                'workloads': entry['workloads'],
                'hardware': entry['hardware'],
                'priorities': entry['priorities'],
                'name': {
                    language: entry['translations'][language]['name']
                    for language in ('zh', 'en')
                },
                'reason': {
                    language: entry['translations'][language]['selection_reason']
                    for language in ('zh', 'en')
                },
                'limitation': {
                    language: entry['translations'][language]['primary_limitation']
                    for language in ('zh', 'en')
                },
            }
            for entry in entries
            if entry.get('selectable', True)
        ],
    }


def _render_page(
    environment: Environment,
    template_name: str,
    destination: Path,
    context: dict[str, Any],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = environment.get_template(template_name).render(**context)
    destination.write_text(rendered.rstrip() + '\n', encoding='utf-8')


def build(output_dir: Path) -> dict[str, Any]:
    """Build a complete product-site directory and return its manifest."""
    output_dir = Path(output_dir).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.funasr-product-build-', dir=output_dir.parent))
    try:
        legacy = SITE_ROOT / 'legacy'
        if legacy.is_dir():
            shutil.copytree(legacy, stage, dirs_exist_ok=True)
            for source in sorted(legacy.rglob('*.html')):
                html = source.read_text(encoding='utf-8')
                if not BeautifulSoup(html, 'html.parser').select_one('nav.nav'):
                    continue
                relative = source.relative_to(legacy)
                language = 'en' if relative.parts[0] == 'en' else 'zh'
                destination = stage / relative
                destination.write_text(
                    normalize_document(html, legacy_route(relative), language),
                    encoding='utf-8',
                )

        registry = load_registry(SITE_ROOT / 'data' / 'deployments.json')
        errors = validate_registry(registry)
        if errors:
            raise ValueError('invalid deployment registry:\n' + '\n'.join(errors))
        navigation = json.loads(
            (SITE_ROOT / 'data' / 'navigation.json').read_text(encoding='utf-8')
        )
        assets, asset_hashes = _copy_hashed_assets(stage)
        environment = Environment(
            loader=FileSystemLoader(SITE_ROOT / 'templates'),
            autoescape=select_autoescape(('html', 'xml')),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        pages: list[dict[str, str]] = []
        language_copy = {
            'zh': {
                'home_title': 'FunASR - 可私有化部署的语音智能基础设施',
                'home_description': '选择适合 GPU 高吞吐、实时流式、OpenAI 兼容 API、CPU 与边缘设备的 FunASR 部署方案。',
                'deploy_title': '部署中心 - FunASR',
                'deploy_description': '按工作负载、硬件和优先级选择可验证的 FunASR 工业部署路径。',
                'detail_suffix': '工业部署 - FunASR',
                'benchmarks_title': '可复现实测 - FunASR',
                'benchmarks_description': '核对 FunASR 公开性能记录的硬件、音频、设置、计时口径、来源与适用限制。',
                'not_found_title': '页面未找到 - FunASR',
                'not_found_description': '返回 FunASR 部署中心、文档或首页。',
            },
            'en': {
                'home_title': 'FunASR - Private-deployment speech infrastructure',
                'home_description': 'Choose a verified FunASR path for GPU throughput, realtime streaming, OpenAI-compatible APIs, CPUs, and edge devices.',
                'deploy_title': 'Deployment center - FunASR',
                'deploy_description': 'Choose an evidence-backed FunASR production path by workload, hardware, and priority.',
                'detail_suffix': 'industrial deployment - FunASR',
                'benchmarks_title': 'Reproducible measurements - FunASR',
                'benchmarks_description': 'Review hardware, audio, settings, timing scope, source, and qualifications for public FunASR performance records.',
                'not_found_title': 'Page not found - FunASR',
                'not_found_description': 'Return to the FunASR deployment center, documentation, or home page.',
            },
        }
        selector_payload = _selector_payload(registry['deployments'])

        for language, route, peer_route in (
            ('zh', '/', '/en/'),
            ('en', '/en/', '/'),
        ):
            context = _page_context(
                language=language,
                route=route,
                peer_route=peer_route,
                title=language_copy[language]['home_title'],
                description=language_copy[language]['home_description'],
                date_modified=registry['verified'],
                navigation=navigation,
                assets=assets,
            )
            context.update({
                'deployments': registry['deployments'],
                'selector_payload': selector_payload,
                'verified': registry['verified'],
            })
            _render_page(environment, 'home.html', route_path(stage, route), context)
            pages.append({
                'route': route,
                'language': language,
                'canonical': context['canonical'],
                'hreflang': context['peer_canonical'],
            })

        for language, route, peer_route in (
            ('zh', '/deploy/', '/en/deploy/'),
            ('en', '/en/deploy/', '/deploy/'),
        ):
            context = _page_context(
                language=language,
                route=route,
                peer_route=peer_route,
                title=language_copy[language]['deploy_title'],
                description=language_copy[language]['deploy_description'],
                date_modified=registry['verified'],
                navigation=navigation,
                assets=assets,
            )
            context.update({'deployments': registry['deployments'], 'verified': registry['verified']})
            _render_page(environment, 'deploy-index.html', route_path(stage, route), context)
            pages.append({
                'route': route,
                'language': language,
                'canonical': context['canonical'],
                'hreflang': context['peer_canonical'],
            })

        for entry in registry['deployments']:
            for language in ('zh', 'en'):
                route = entry['routes'][language]
                peer_language = 'en' if language == 'zh' else 'zh'
                peer_route = entry['routes'][peer_language]
                translation = entry['translations'][language]
                context = _page_context(
                    language=language,
                    route=route,
                    peer_route=peer_route,
                    title=f"{translation['name']} - {language_copy[language]['detail_suffix']}",
                    description=translation['summary'],
                    date_modified=entry['tested']['verified'],
                    navigation=navigation,
                    assets=assets,
                )
                context.update({'entry': entry, 'verified': registry['verified']})
                _render_page(
                    environment,
                    'deploy-detail.html',
                    route_path(stage, route),
                    context,
                )
                pages.append({
                    'route': route,
                    'language': language,
                    'canonical': context['canonical'],
                    'hreflang': context['peer_canonical'],
                })

        for language, route, peer_route in (
            ('zh', '/benchmarks.html', '/en/benchmarks.html'),
            ('en', '/en/benchmarks.html', '/benchmarks.html'),
        ):
            benchmarks = []
            for entry in registry['deployments']:
                for record in entry['benchmarks']:
                    localized_record = dict(record)
                    localized_record['deployment_id'] = entry['id']
                    localized_record['deployment_name'] = entry['translations'][language]['name']
                    localized_record['deployment_route'] = entry['routes'][language]
                    benchmarks.append(localized_record)
            context = _page_context(
                language=language,
                route=route,
                peer_route=peer_route,
                title=language_copy[language]['benchmarks_title'],
                description=language_copy[language]['benchmarks_description'],
                date_modified=registry['verified'],
                navigation=navigation,
                assets=assets,
            )
            context.update({'benchmarks': benchmarks, 'verified': registry['verified']})
            _render_page(environment, 'benchmarks.html', route_path(stage, route), context)
            pages.append({
                'route': route,
                'language': language,
                'canonical': context['canonical'],
                'hreflang': context['peer_canonical'],
            })

        context = _page_context(
            language='zh',
            route='/404.html',
            peer_route='/en/deploy/',
            title=language_copy['zh']['not_found_title'],
            description=language_copy['zh']['not_found_description'],
            date_modified=registry['verified'],
            navigation=navigation,
            assets=assets,
        )
        _render_page(environment, '404.html', route_path(stage, '/404.html'), context)
        pages.append({
            'route': '/404.html',
            'language': 'zh',
            'canonical': context['canonical'],
            'hreflang': context['peer_canonical'],
        })

        last_modified_by_route = {
            entry['routes'][language]: entry['tested']['verified']
            for entry in registry['deployments']
            for language in ('zh', 'en')
        }
        _write_sitemap(stage, last_modified_by_route)

        manifest = {
            'schema_version': 1,
            'verified': registry['verified'],
            'pages': pages,
            'assets': dict(sorted(asset_hashes.items())),
        }
        (stage / 'deployment-manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        stage.replace(output_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.output)
    print(f"built {len(manifest['pages'])} product pages in {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
