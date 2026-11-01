"""Legacy URLs must lead to maintained, model-specific instructions."""

from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('source,targets', (
    ('docs/reference/build_task.md', ('../model_registration.md', '../training.md')),
    ('docs/reference/application.md', ('../moss_transcribe_diarize.md', '../use_case_showcase.md')),
    ('docs/reference/FQA.md', ('../troubleshooting.md', '../moss_transcribe_diarize.md',
                              '../installation/installation.md')),
))
def test_legacy_reference_pages_link_to_current_guides(source, targets):
    path = ROOT / source
    content = path.read_text()
    links = re.findall(r'\]\(([^)]+)\)', content)
    for target in targets:
        assert target in links
        assert (path.parent / target).is_file()


def test_removed_task_architecture_is_not_an_executable_tutorial():
    content = (ROOT / 'docs/reference/build_task.md').read_text()
    assert content.startswith('# Build custom tasks\n')
    assert '```python' not in content
    assert 'migration' in content.lower()
    assert 'funasr/tasks/abs_task.py' not in content


def test_legacy_faq_preserves_model_boundaries():
    content = (ROOT / 'docs/reference/FQA.md').read_text()
    assert 'MOSS' in content
    assert 'anonymous' in content
    assert 'external `vad_model` or `spk_model`' in content
    assert 'Use a model pipeline that includes both VAD and speaker models:' not in content
    assert 'retry after removing the `funasr-cache` Docker volume' not in content
    assert '## Speaker diarization has no speaker labels' in content
    assert '## Can I use a speaker model other than cam++?' in content


@pytest.mark.parametrize('source', (
    'model_zoo/huggingface_models.md',
    'model_zoo/modelscope_models.md',
    'model_zoo/modelscope_models_zh.md',
))
def test_hub_catalogues_do_not_assign_one_license_to_all_models(source):
    content = (ROOT / source).read_text()
    intro = content[:content.index('### ')]
    assert '../MODEL_LICENSE' in intro
    assert '../LICENSE' in intro
    assert 'OpenMOSS' in intro
    assert '-  Apache License 2.0' not in intro


def test_ascend_guide_scopes_historical_model_and_environment_evidence():
    content = (ROOT / 'docs/ascend_npu.md').read_text(encoding='utf-8')
    intro = content.split('## 1.', 1)[0]
    for identity in ('310P3', 'CANN 9.1.0-beta.1', 'torch 2.10.0+cpu',
                     'torch_npu 2.10.0', 'funasr 1.3.14', 'fsmn-vad',
                     'paraformer', 'ct-transformer', 'cam++'):
        assert identity in intro
    assert 'https://github.com/modelscope/FunASR/pull/3708' in intro
    assert '\u4e0d\u8986\u76d6 Nano/Qwen3' in intro
    assert 'CPU' in intro and '\u4fee\u8ba2\u7248' in intro
    assert '\u4e0a\u5b8c\u6574\u9a8c\u8bc1' not in intro


def test_ascend_pooling_and_latency_are_not_universal_npu_guarantees():
    content = (ROOT / 'docs/ascend_npu.md').read_text(encoding='utf-8')
    pooling = content.split('## 3.', 1)[1].split('## 4.', 1)[0]
    assert '\u4ec5 CPU \u8bed\u4e49\u68c0\u67e5' in pooling
    assert '\u53e6\u884c\u9a8c\u8bc1' in pooling
    assert '\u4ec5\u7528 NPU \u652f\u6301\u7684\u57fa\u7840\u7b97\u5b50' not in pooling
    behavior = content.split('## 5.', 1)[1]
    assert '\u51b7\u542f\u52a8' in behavior
    assert '\u9884\u70ed\u540e' in behavior
    assert '\u8f93\u51fa\u6b63\u786e\u6027' in behavior
    assert '\u4e0d\u80fd\u4ec5\u51ed' in behavior
    assert '\u5c5e\u6b63\u5e38\u73b0\u8c61\uff0c\u975e\u5361\u6b7b' not in behavior
    assert 'fsmn-vad / paraformer / ct-transformer \u53ef\u76f4\u63a5\u8fd0\u884c' not in behavior
