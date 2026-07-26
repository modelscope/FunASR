"""Deterministic deployment recommendation scoring."""

from __future__ import annotations

from typing import Any


SUPPORTED_VALUES = {
    'workload': {'batch', 'realtime', 'private-api', 'edge'},
    'hardware': {'nvidia-gpu', 'cpu', 'desktop-edge-gpu', 'kubernetes'},
    'priority': {'throughput', 'latency', 'portability', 'compatibility'},
}
MATCH_WEIGHTS = {
    'workload': 4,
    'hardware': 3,
    'priority': 2,
}


def _validate_choice(name: str, value: str) -> None:
    if value not in SUPPORTED_VALUES[name]:
        raise ValueError(f'Unsupported {name}: {value}')


def recommend(
    entries: list[dict[str, Any]],
    workload: str,
    hardware: str,
    priority: str,
) -> dict[str, Any]:
    """Return the highest-scoring registry entry with a stable tie break."""
    choices = {
        'workload': workload,
        'hardware': hardware,
        'priority': priority,
    }
    for name, value in choices.items():
        _validate_choice(name, value)

    scored: list[tuple[int, int, str, dict[str, Any]]] = []
    for entry in entries:
        if entry.get('selectable', True) is False:
            continue
        score = 0
        score += MATCH_WEIGHTS['workload'] * (workload in entry.get('workloads', []))
        score += MATCH_WEIGHTS['hardware'] * (hardware in entry.get('hardware', []))
        score += MATCH_WEIGHTS['priority'] * (priority in entry.get('priorities', []))
        scored.append((
            -score,
            int(entry.get('selector_rank', 10_000)),
            str(entry.get('id', '')),
            entry,
        ))

    if not scored:
        raise ValueError('No selectable deployment entries')

    negative_score, _, _, entry = min(scored)
    result = dict(entry)
    result['match_score'] = -negative_score
    result['selection_reason'] = {
        language: entry['translations'][language]['selection_reason']
        for language in ('zh', 'en')
    }
    result['primary_limitation'] = {
        language: entry['translations'][language]['primary_limitation']
        for language in ('zh', 'en')
    }
    return result
