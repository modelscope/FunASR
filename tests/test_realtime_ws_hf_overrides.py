import pytest

from test_realtime_ws_service import load_service_module


@pytest.mark.parametrize("value", ["[]", "[1]", '"model"', "1", "true", "false", "null"])
def test_cli_rejects_non_object_hf_overrides(value, capsys):
    parser = load_service_module().build_arg_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--hf-overrides", value])

    assert exc.value.code == 2
    assert "JSON object" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["{", "{bad}", '{"architectures":}'])
def test_cli_rejects_malformed_hf_overrides(value):
    parser = load_service_module().build_arg_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--hf-overrides", value])

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("{}", {}),
        ('{"architectures":["Qwen2ForCausalLM"]}', {"architectures": ["Qwen2ForCausalLM"]}),
        ('{"rope_scaling":{"factor":2.0},"use_cache":false}',
         {"rope_scaling": {"factor": 2.0}, "use_cache": False}),
    ],
)
def test_cli_preserves_hf_override_objects(value, expected):
    parser = load_service_module().build_arg_parser()

    args = parser.parse_args(["--hf-overrides", value])

    assert args.hf_overrides == expected
