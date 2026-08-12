from __future__ import annotations

import sys
from pathlib import Path

import pytest


SITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE_ROOT))

from registry import load_registry  # noqa: E402
from selector import recommend  # noqa: E402


@pytest.fixture
def entries():
    return load_registry(SITE_ROOT / 'data' / 'deployments.json')['deployments']


@pytest.mark.parametrize(
    ('workload', 'hardware', 'priority', 'expected'),
    [
        ('batch', 'nvidia-gpu', 'throughput', 'vllm'),
        ('edge', 'cpu', 'portability', 'llama-cpp'),
        ('realtime', 'cpu', 'compatibility', 'sensevoice-native-server'),
        ('private-api', 'kubernetes', 'compatibility', 'containers'),
        ('realtime', 'nvidia-gpu', 'latency', 'realtime'),
    ],
)
def test_recommendation_matrix(workload, hardware, priority, expected, entries):
    result = recommend(entries, workload, hardware, priority)

    assert result['id'] == expected
    assert result['match_score'] == 9
    assert set(result['selection_reason']) == {'zh', 'en'}
    assert set(result['primary_limitation']) == {'zh', 'en'}


def test_unknown_selector_value_is_rejected(entries):
    with pytest.raises(ValueError, match='Unsupported hardware: tpu'):
        recommend(entries, 'batch', 'tpu', 'throughput')


def test_registry_order_does_not_change_tie_break(entries):
    forward = recommend(entries, 'private-api', 'kubernetes', 'compatibility')
    reverse = recommend(list(reversed(entries)), 'private-api', 'kubernetes', 'compatibility')

    assert forward['id'] == reverse['id'] == 'containers'
