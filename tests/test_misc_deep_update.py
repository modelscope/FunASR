from funasr.utils.misc import deep_update


def test_none_replaced_by_dict():
    original = {"ctc_conf": None}

    deep_update(original, {"ctc_conf": {"dropout_rate": 0.1}})

    assert original == {"ctc_conf": {"dropout_rate": 0.1}}


def test_string_replaced_by_dict():
    original = {"ctc_conf": "disabled"}

    deep_update(original, {"ctc_conf": {"dropout_rate": 0.1}})

    assert original == {"ctc_conf": {"dropout_rate": 0.1}}


def test_nested_dict_merge_keeps_untouched_keys():
    original = {"ctc_conf": {"dropout_rate": 0.0, "reduce": True}, "encoder": "test"}

    deep_update(original, {"ctc_conf": {"dropout_rate": 0.1}})

    assert original == {
        "ctc_conf": {"dropout_rate": 0.1, "reduce": True},
        "encoder": "test",
    }


def test_empty_dict_replaces_value():
    original = {"ctc_conf": {"dropout_rate": 0.1}}

    deep_update(original, {"ctc_conf": {}})

    assert original == {"ctc_conf": {}}
