from collections import OrderedDict

import pytest

from funasr.auto.auto_model import AutoModel
from funasr.utils.misc import deep_update


@pytest.mark.parametrize("old", [None, "disabled", 0, 0.5, False, ["old"]])
def test_non_mapping_replaced_in_place(old):
    original = {"ctc_conf": old, "untouched": True}
    replacement = {"dropout_rate": 0.1}
    assert deep_update(original, {"ctc_conf": replacement}) is None
    assert original == {"ctc_conf": replacement, "untouched": True}
    assert original["ctc_conf"] is replacement


def test_missing_key_and_recursive_merge_preserve_contract():
    nested = OrderedDict(keep=1, change=2)
    original = {"nested": nested}
    added = {"new": 3}
    deep_update(original, {"nested": {"change": 4}, "added": added})
    assert original == {"nested": {"keep": 1, "change": 4}, "added": {"new": 3}}
    assert original["nested"] is nested
    assert original["added"] is added


@pytest.mark.parametrize("replacement", [{}, None, "disabled", 0, []])
def test_dict_replaced_by_empty_or_scalar(replacement):
    original = {"nested": {"old": True}}
    deep_update(original, {"nested": replacement})
    assert original["nested"] is replacement


def test_auto_model_runtime_config_replaces_none_without_moving_device():
    kwargs = {"device": "cuda:0", "decoding": None, "encoder": {"keep": 1}}
    AutoModel._merge_runtime_config(
        kwargs, {"device": "cpu", "decoding": {"beam_size": 2}, "encoder": {"new": 3}}
    )
    assert kwargs == {
        "device": "cuda:0",
        "decoding": {"beam_size": 2},
        "encoder": {"keep": 1, "new": 3},
    }
