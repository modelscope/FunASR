from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


SITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE_ROOT))

from registry import deployment_pairs, load_registry, validate_registry  # noqa: E402


REGISTRY = SITE_ROOT / 'data' / 'deployments.json'
EXPECTED_IDS = {
    'vllm',
    'llama-cpp',
    'openai-api',
    'realtime',
    'containers',
    'cpu-runtime',
    'production',
}


@pytest.fixture
def valid_registry():
    return load_registry(REGISTRY)


def test_checked_in_registry_is_valid(valid_registry):
    assert validate_registry(valid_registry) == []


def test_registry_has_all_product_routes(valid_registry):
    assert {item['id'] for item in valid_registry['deployments']} == EXPECTED_IDS


def test_language_pairs_have_identical_fields(valid_registry):
    assert all(set(zh) == set(en) for zh, en in deployment_pairs(valid_registry))


def test_production_entry_requires_evidence(valid_registry):
    data = copy.deepcopy(valid_registry)
    entry = next(item for item in data['deployments'] if item['maturity'] == 'production-verified')
    del entry['evidence']

    assert validate_registry(data) == [
        f"{entry['id']}: production-verified entry requires evidence"
    ]


def test_production_entry_requires_verification_contract(valid_registry):
    data = copy.deepcopy(valid_registry)
    entry = next(item for item in data['deployments'] if item['maturity'] == 'production-verified')
    del entry['tested']['verified']

    assert validate_registry(data) == [
        f"{entry['id']}: production-verified entry requires tested.verified"
    ]


def test_duplicate_route_is_rejected(valid_registry):
    data = copy.deepcopy(valid_registry)
    data['deployments'][1]['routes']['zh'] = data['deployments'][0]['routes']['zh']

    errors = validate_registry(data)

    assert any('duplicate route' in error for error in errors)


def test_evidence_must_be_https(valid_registry):
    data = copy.deepcopy(valid_registry)
    data['deployments'][0]['evidence'][0]['url'] = 'http://example.com/evidence'

    assert any('evidence URL must use https' in error for error in validate_registry(data))


def test_benchmark_requires_reproducibility_fields(valid_registry):
    data = copy.deepcopy(valid_registry)
    entry = data['deployments'][0]
    entry['benchmarks'] = [{
        'model': 'Fun-ASR-Nano-2512',
        'runtime': 'vLLM',
        'hardware': 'NVIDIA GPU',
    }]

    errors = validate_registry(data)

    assert any('benchmark 0 requires workload' in error for error in errors)
