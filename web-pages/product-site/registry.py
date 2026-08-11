"""Deployment registry loading and evidence validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


MATURITY_VALUES = {
    'production-verified',
    'community-verified',
    'experimental',
}
LANGUAGES = ('zh', 'en')
TRANSLATION_FIELDS = {
    'name',
    'summary',
    'fit',
    'not_fit',
    'selection_reason',
    'primary_limitation',
    'status_label',
    'operations',
    'security',
    'troubleshooting',
}
PRODUCTION_FIELDS = (
    ('tested.verified', ('tested', 'verified')),
    ('tested.funasr', ('tested', 'funasr')),
    ('tested.runtime', ('tested', 'runtime')),
    ('commands.smoke', ('commands', 'smoke')),
)
BENCHMARK_FIELDS = (
    'model',
    'runtime',
    'hardware',
    'workload',
    'audio',
    'settings',
    'timing_scope',
    'result',
    'qualification',
    'source',
    'verified',
)
DOWNLOAD_FIELDS = ('operating_system', 'architecture', 'backend', 'archive', 'url', 'sha256')


def load_registry(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON deployment registry."""
    with path.open(encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError('deployment registry root must be an object')
    return data


def deployment_pairs(data: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return Chinese and English translation pairs in registry order."""
    return [
        (entry.get('translations', {}).get('zh', {}), entry.get('translations', {}).get('en', {}))
        for entry in data.get('deployments', [])
    ]


def _nested_value(entry: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = entry
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _is_https_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == 'https' and bool(parsed.netloc)


def validate_registry(data: dict[str, Any]) -> list[str]:
    """Return stable content errors without raising for malformed entries."""
    errors: list[str] = []
    deployments = data.get('deployments')
    if not isinstance(deployments, list):
        return ['registry: deployments must be a list']

    seen_ids: set[str] = set()
    seen_routes: set[str] = set()

    for index, entry in enumerate(deployments):
        if not isinstance(entry, dict):
            errors.append(f'entry {index}: must be an object')
            continue

        entry_id = entry.get('id')
        label = entry_id if isinstance(entry_id, str) and entry_id else f'entry {index}'
        if not isinstance(entry_id, str) or not entry_id:
            errors.append(f'{label}: id is required')
        elif entry_id in seen_ids:
            errors.append(f'{label}: duplicate id {entry_id}')
        else:
            seen_ids.add(entry_id)

        maturity = entry.get('maturity')
        if maturity not in MATURITY_VALUES:
            errors.append(f'{label}: invalid maturity {maturity!r}')

        routes = entry.get('routes')
        if not isinstance(routes, dict):
            errors.append(f'{label}: routes must be an object')
        else:
            for language in LANGUAGES:
                route = routes.get(language)
                if not isinstance(route, str) or not route.startswith('/'):
                    errors.append(f'{label}: routes.{language} must be an absolute route')
                elif route in seen_routes:
                    errors.append(f'{label}: duplicate route {route}')
                else:
                    seen_routes.add(route)

        translations = entry.get('translations')
        if not isinstance(translations, dict):
            errors.append(f'{label}: translations must be an object')
        else:
            translation_keys = []
            for language in LANGUAGES:
                content = translations.get(language)
                if not isinstance(content, dict):
                    errors.append(f'{label}: translations.{language} must be an object')
                    translation_keys.append(set())
                    continue
                translation_keys.append(set(content))
                missing = sorted(TRANSLATION_FIELDS - set(content))
                for field in missing:
                    errors.append(f'{label}: translations.{language}.{field} is required')
                if not content.get('primary_limitation'):
                    errors.append(f'{label}: translations.{language}.primary_limitation is required')
            if len(translation_keys) == 2 and translation_keys[0] != translation_keys[1]:
                errors.append(f'{label}: translation fields must match')

        evidence = entry.get('evidence')
        if maturity == 'production-verified' and not evidence:
            errors.append(f'{label}: production-verified entry requires evidence')
        if evidence is not None:
            if not isinstance(evidence, list):
                errors.append(f'{label}: evidence must be a list')
            else:
                for evidence_index, item in enumerate(evidence):
                    if not isinstance(item, dict) or not _is_https_url(item.get('url')):
                        errors.append(
                            f'{label}: evidence URL must use https (item {evidence_index})'
                        )

        if maturity == 'production-verified':
            for field_name, field_path in PRODUCTION_FIELDS:
                if not _nested_value(entry, field_path):
                    errors.append(
                        f'{label}: production-verified entry requires {field_name}'
                    )

        downloads = entry.get('downloads', [])
        if not isinstance(downloads, list):
            errors.append(f'{label}: downloads must be a list')
        else:
            for download_index, download in enumerate(downloads):
                if not isinstance(download, dict):
                    errors.append(f'{label}: download {download_index} must be an object')
                    continue
                for field in DOWNLOAD_FIELDS:
                    if not download.get(field):
                        errors.append(
                            f'{label}: download {download_index} requires {field}'
                        )
                if not _is_https_url(download.get('url')):
                    errors.append(
                        f'{label}: download URL must use https (item {download_index})'
                    )
                digest = download.get('sha256')
                if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                    errors.append(
                        f'{label}: download SHA-256 must be 64 lowercase hex characters '
                        f'(item {download_index})'
                    )

        benchmarks = entry.get('benchmarks', [])
        if not isinstance(benchmarks, list):
            errors.append(f'{label}: benchmarks must be a list')
        else:
            for benchmark_index, benchmark in enumerate(benchmarks):
                if not isinstance(benchmark, dict):
                    errors.append(f'{label}: benchmark {benchmark_index} must be an object')
                    continue
                for field in BENCHMARK_FIELDS:
                    if not benchmark.get(field):
                        errors.append(
                            f'{label}: benchmark {benchmark_index} requires {field}'
                        )

    return errors
