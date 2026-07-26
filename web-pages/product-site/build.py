"""Build the dependency-free FunASR product site."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from registry import load_registry, validate_registry
from selector import MATCH_WEIGHTS


SITE_ROOT = Path(__file__).resolve().parent
BASE_URL = 'https://www.funasr.com'


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
        'navigation': _navigation(navigation, language),
        'assets': assets,
        'github_url': 'https://github.com/modelscope/FunASR',
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
                'not_found_title': '页面未找到 - FunASR',
                'not_found_description': '返回 FunASR 部署中心、文档或首页。',
            },
            'en': {
                'home_title': 'FunASR - Private-deployment speech infrastructure',
                'home_description': 'Choose a verified FunASR path for GPU throughput, realtime streaming, OpenAI-compatible APIs, CPUs, and edge devices.',
                'deploy_title': 'Deployment center - FunASR',
                'deploy_description': 'Choose an evidence-backed FunASR production path by workload, hardware, and priority.',
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

        context = _page_context(
            language='zh',
            route='/404.html',
            peer_route='/en/deploy/',
            title=language_copy['zh']['not_found_title'],
            description=language_copy['zh']['not_found_description'],
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
