"""Normalize legacy page navigation and language metadata without rewriting content."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

from bs4 import BeautifulSoup


SITE_ROOT = Path(__file__).resolve().parent
BASE_URL = 'https://www.funasr.com'
METADATA_START = '<!-- product-site-metadata:start -->'
METADATA_END = '<!-- product-site-metadata:end -->'
NAVIGATION = json.loads(
    (SITE_ROOT / 'data' / 'navigation.json').read_text(encoding='utf-8')
)


def _peer_route(route: str, language: str) -> str:
    if language == 'zh':
        if route == '/':
            return '/en/'
        return f'/en{route}'
    if route == '/en/':
        return '/'
    if not route.startswith('/en/'):
        raise ValueError(f'English legacy route must start with /en/: {route}')
    return route[3:]


def _canonical_routes(route: str, language: str) -> tuple[str, str]:
    if route in {'/llama-cpp.html', '/en/llama-cpp.html'}:
        if language == 'zh':
            return '/deploy/llama-cpp.html', '/en/deploy/llama-cpp.html'
        return '/en/deploy/llama-cpp.html', '/deploy/llama-cpp.html'
    return route, _peer_route(route, language)


def _navigation_markup(language: str, peer_route: str) -> str:
    label_key = 'zh' if language == 'zh' else 'en'
    href_key = 'href_zh' if language == 'zh' else 'href_en'
    links = ''.join(
        f'<a href="{escape(item[href_key], quote=True)}">{escape(item[label_key])}</a>'
        for item in NAVIGATION['items']
    )
    peer_label = 'EN' if language == 'zh' else '中文'
    return (
        '<nav class="nav" data-product-navigation="true"><div class="container">'
        f'<a href="{"/" if language == "zh" else "/en/"}" class="nav-logo">'
        'Fun<span>ASR</span></a>'
        f'<div class="nav-links">{links}</div>'
        f'<a href="{escape(peer_route, quote=True)}" data-language-peer '
        f'style="font-size:0.83rem;font-weight:500;color:var(--text-soft)">{peer_label}</a>'
        '<a href="/go/github" target="_blank" '
        'rel="noopener" class="nav-btn">GitHub</a></div></nav>'
    )


def _replace_navigation(html: str, replacement: str) -> str:
    start = re.search(
        r'<nav\b(?=[^>]*\bclass\s*=\s*["\'][^"\']*\bnav\b)[^>]*>',
        html,
        re.IGNORECASE,
    )
    if start is None:
        raise ValueError('Unable to locate legacy navigation in source')
    end = re.search(r'</nav\s*>', html[start.end():], re.IGNORECASE)
    if end is None:
        raise ValueError('Unable to locate closing legacy navigation in source')
    end_position = start.end() + end.end()
    return html[:start.start()] + replacement + html[end_position:]


def _remove_metadata_links(html: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        fragment = match.group(0)
        link = BeautifulSoup(fragment, 'html.parser').find('link')
        if link is None:
            return fragment
        rel = {str(value).lower() for value in link.get('rel', [])}
        href = str(link.get('href', ''))
        if rel.intersection({'canonical', 'alternate'}) or 'fonts.googleapis.com' in href:
            return ''
        return fragment

    return re.sub(r'<link\b[^>]*>', replacement, html, flags=re.IGNORECASE)


def _remove_excluded_runtime_dependencies(html: str) -> str:
    return re.sub(
        r'<script\b(?=[^>]*\bsrc\s*=\s*["\']/stats/tracker\.js(?:\?[^"\']*)?["\'])'
        r'[^>]*>\s*</script>',
        '',
        html,
        flags=re.IGNORECASE,
    )


def _metadata_markup(route: str, language: str) -> str:
    canonical_route, peer_route = _canonical_routes(route, language)
    peer_language = 'en' if language == 'zh' else 'zh-CN'
    current_language = 'zh-CN' if language == 'zh' else 'en'
    return '\n'.join((
        METADATA_START,
        f'<link rel="canonical" href="{BASE_URL}{canonical_route}">',
        f'<link rel="alternate" hreflang="{peer_language}" href="{BASE_URL}{peer_route}">',
        f'<link rel="alternate" hreflang="{current_language}" href="{BASE_URL}{canonical_route}">',
        f'<link rel="alternate" hreflang="x-default" href="{BASE_URL}{peer_route if language == "zh" else canonical_route}">',
        METADATA_END,
    ))


def normalize_document(html: str, route: str, language: str) -> str:
    """Return a legacy page with product navigation and canonical metadata."""
    if language not in {'zh', 'en'}:
        raise ValueError(f'Unsupported language: {language}')
    if not route.startswith('/'):
        raise ValueError(f'Route must start with /: {route}')

    soup = BeautifulSoup(html, 'html.parser')
    navigation = soup.select('nav.nav')
    if len(navigation) != 1:
        raise ValueError(f'Expected one legacy navigation shell, found {len(navigation)}')
    if soup.head is None:
        raise ValueError('Expected a legacy document head')

    peer_route = _peer_route(route, language)
    normalized = _replace_navigation(
        html,
        _navigation_markup(language, peer_route),
    )
    normalized = _remove_excluded_runtime_dependencies(normalized)

    marker_pattern = re.compile(
        re.escape(METADATA_START) + r'.*?' + re.escape(METADATA_END),
        re.DOTALL,
    )
    metadata = _metadata_markup(route, language)
    if marker_pattern.search(normalized):
        return marker_pattern.sub(metadata, normalized, count=1)

    normalized = _remove_metadata_links(normalized)

    head_close = normalized.lower().find('</head>')
    if head_close < 0:
        raise ValueError('Expected a closing legacy document head')
    return normalized[:head_close] + metadata + '\n' + normalized[head_close:]
