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
    'sensevoice-tensorrt',
    'llama-cpp',
    'sensevoice-native-server',
    'audio-cpp',
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


def test_audio_cpp_contract_tracks_stable_nano_and_candidate_sensevoice(valid_registry):
    entry = next(item for item in valid_registry['deployments'] if item['id'] == 'audio-cpp')
    llama_cpp = next(item for item in valid_registry['deployments'] if item['id'] == 'llama-cpp')

    assert valid_registry['verified'] == '2026-08-13'
    assert entry['maturity'] == 'community-verified'
    assert entry['selector_rank'] > llama_cpp['selector_rank']
    assert entry['tested'] == {
        'funasr': 'Fun-ASR-Nano-2512 + SenseVoice-Small',
        'runtime': 'audio.cpp@1778b23a + SenseVoice candidate@b748ca5',
        'verified': '2026-08-13',
    }
    assert entry['models'] == ['Fun-ASR-Nano-2512', 'SenseVoice-Small']
    assert 'Buffered streaming CLI/SSE' in entry['interfaces']
    assert any(
        'git checkout b748ca509adc16c15aff44f76456fd47b257c933' in command
        for command in entry['commands']['install']
    )
    assert any(
        'model_manager_v2.py install fun_asr_nano' in command
        for command in entry['commands']['install']
    )
    assert any(
        'model_manager_v2.py install sensevoice_small_q8' in command
        for command in entry['commands']['install']
    )
    build_command = next(command for command in entry['commands']['install'] if '--model-set custom' in command)
    assert build_command.startswith('bash scripts/build_linux.sh ')
    assert '--models fun_asr_nano,sense_asr' in build_command
    assert any('--family sense_asr' in command for command in entry['commands']['launch'])
    assert any(
        '--mode streaming' in command and 'audio_chunk_duration_sec=5' in command
        for command in entry['commands']['launch']
    )
    assert any('/v1/audio/transcriptions' in command for command in entry['commands']['smoke'])
    assert any('/health' in command for command in entry['commands']['health'])
    assert any('1778b23a5f6a4951c788e4bb0e7baa04f20012a2' in item['url'] for item in entry['evidence'])
    assert any('ce72677f84900f0dc57f498ace253bfb3c9155b6' in item['url'] for item in entry['evidence'])
    assert any('/pull/219' in item['url'] for item in entry['evidence'])
    assert any(
        '5c3fcfe748a8714216bc135476d5863084fddb72' in item['url']
        for item in entry['evidence']
    )
    assert any(
        '4dedf169f625437fb336f2959674f399819729a765e184128c0e25a6e16ff0ec'
        in item['label']
        for item in entry['evidence']
    )
    assert any(
        '4dedf169f625437fb336f2959674f399819729a765e184128c0e25a6e16ff0ec'
        in benchmark['result']
        for benchmark in entry['benchmarks']
    )
    assert 'candidate' in entry['translations']['en']['primary_limitation'].lower()
    assert 'timestamp' in entry['translations']['en']['primary_limitation'].lower()


def test_sensevoice_tensorrt_contract_tracks_merged_native_runtime(valid_registry):
    entry = next(
        item for item in valid_registry['deployments']
        if item['id'] == 'sensevoice-tensorrt'
    )
    vllm = next(item for item in valid_registry['deployments'] if item['id'] == 'vllm')
    llama_cpp = next(item for item in valid_registry['deployments'] if item['id'] == 'llama-cpp')

    assert entry['maturity'] == 'production-verified'
    assert vllm['selector_rank'] < entry['selector_rank'] < llama_cpp['selector_rank']
    assert entry['tested'] == {
        'funasr': 'main@6408aaa9',
        'runtime': 'TensorRT 10.0.1 / Triton 24.05',
        'verified': '2026-08-04',
    }
    assert entry['models'] == ['SenseVoiceSmall']
    assert entry['hardware'] == ['nvidia-gpu', 'kubernetes']
    assert 'Triton gRPC/HTTP' in entry['interfaces']
    assert any('git checkout 6408aaa9' in command for command in entry['commands']['install'])
    assert any('quantize=False' in command for command in entry['commands']['install'])
    assert any(
        'chn_jpn_yue_eng_ko_spectok.bpe.model' in command
        and 'aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8' in command
        for command in entry['commands']['install']
    )
    assert any(
        'build_sensevoice_tensorrt.py' in command
        and '--max-batch 16' in command
        and '--max-frames 4096' in command
        for command in entry['commands']['launch']
    )
    assert any(
        command.startswith('cd runtime/triton_gpu && tritonserver')
        for command in entry['commands']['launch']
    )
    assert any('/v2/health/ready' in command for command in entry['commands']['health'])
    assert any('TRANSCRIPTS' in command for command in entry['commands']['smoke'])
    assert any('/pull/3463' in item['url'] for item in entry['evidence'])
    assert any(
        'build_sensevoice_tensorrt.py' in item['url'] for item in entry['evidence']
    )
    assert any(
        '527,504,916 bytes' in benchmark['result']
        and '100% CTC top-1 agreement' in benchmark['result']
        for benchmark in entry['benchmarks']
    )
    limitation = entry['translations']['en']['primary_limitation'].lower()
    assert 'gpu architecture' in limitation
    assert 'tensorrt version' in limitation


def test_llama_cpp_contract_tracks_v020_release_assets(valid_registry):
    entry = next(item for item in valid_registry['deployments'] if item['id'] == 'llama-cpp')

    assert entry['tested'] == {
        'funasr': 'runtime-llamacpp-v0.2.0',
        'runtime': 'llama.cpp@803b7fca',
        'verified': '2026-08-11',
    }
    assert len(entry['downloads']) == 9
    assert {item['archive'] for item in entry['downloads']} == {
        'funasr-llamacpp-linux-arm64.tar.gz',
        'funasr-llamacpp-linux-x64.tar.gz',
        'funasr-llamacpp-linux-x64-avx2.tar.gz',
        'funasr-llamacpp-linux-x64-vulkan.tar.gz',
        'funasr-llamacpp-macos-arm64.tar.gz',
        'funasr-llamacpp-windows-x64.zip',
        'funasr-llamacpp-windows-x64-avx2.zip',
        'funasr-llamacpp-windows-x64-vulkan.zip',
        'funasr-llamacpp-windows-x64-cuda.zip',
    }
    assert all(item['url'].startswith(
        'https://github.com/modelscope/FunASR/releases/download/runtime-llamacpp-v0.2.0/'
    ) for item in entry['downloads'])
    assert all(len(item['sha256']) == 64 for item in entry['downloads'])
    assert any('actions/runs/31458121788' in item['url'] for item in entry['evidence'])
    assert any(
        'download-funasr-model.sh sensevoice ./funasr-gguf f16' in command
        for command in entry['commands']['install']
    )
    assert 'sensevoice-small-f16.gguf' in entry['commands']['launch'][0]
    assert 'AMD' in entry['translations']['en']['primary_limitation']


def test_sensevoice_native_server_contract_tracks_merged_runtime(valid_registry):
    entry = next(
        item for item in valid_registry['deployments']
        if item['id'] == 'sensevoice-native-server'
    )
    llama_cpp = next(item for item in valid_registry['deployments'] if item['id'] == 'llama-cpp')
    audio_cpp = next(item for item in valid_registry['deployments'] if item['id'] == 'audio-cpp')

    assert valid_registry['verified'] == '2026-08-13'
    assert entry['maturity'] == 'production-verified'
    assert llama_cpp['selector_rank'] < entry['selector_rank'] < audio_cpp['selector_rank']
    assert entry['tested'] == {
        'funasr': 'SenseVoice main@b054623c',
        'runtime': 'sensevoice-server@558bd67c',
        'verified': '2026-08-13',
    }
    assert entry['models'] == ['SenseVoiceSmall-GGUF', 'FSMN-VAD-GGUF']
    assert entry['operating_systems'] == ['Linux']
    assert {
        'OpenAI-compatible HTTP',
        'OpenAI realtime WebSocket',
        'SSE',
        'SRT/VTT',
    } <= set(entry['interfaces'])
    assert any(
        'git checkout b054623cca8f015b73ec471dce4f473ac47413da' in command
        for command in entry['commands']['install']
    )
    assert any(
        'download-funasr-model.sh sensevoice' in command
        for command in entry['commands']['install']
    )
    launch = '\n'.join(entry['commands']['launch'])
    for marker in ('sensevoice-server', '--max-connections', '--max-audio-seconds'):
        assert marker in launch
    smoke = '\n'.join(entry['commands']['smoke'])
    for marker in ('/v1/audio/transcriptions', 'response_format=vtt', 'stream_client.py'):
        assert marker in smoke
    assert any('/health' in command for command in entry['commands']['health'])
    assert any('/v1/models' in command for command in entry['commands']['health'])
    evidence_urls = {item['url'] for item in entry['evidence']}
    assert 'https://github.com/QwenAudio/SenseVoice/pull/341' in evidence_urls
    assert 'https://github.com/QwenAudio/SenseVoice/actions/runs/31633730096' in evidence_urls
    limitation = entry['translations']['en']['primary_limitation'].lower()
    assert 'linux' in limitation
    assert 'authentication' in limitation


def test_download_assets_require_https_and_sha256(valid_registry):
    data = copy.deepcopy(valid_registry)
    entry = next(item for item in data['deployments'] if item['id'] == 'llama-cpp')
    entry['downloads'][0]['url'] = 'http://example.com/runtime.tar.gz'
    entry['downloads'][1]['sha256'] = 'not-a-sha256'

    errors = validate_registry(data)

    assert 'llama-cpp: download URL must use https (item 0)' in errors
    assert 'llama-cpp: download SHA-256 must be 64 lowercase hex characters (item 1)' in errors


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
