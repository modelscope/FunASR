import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_UTILS_PATH = REPO_ROOT / "funasr" / "models" / "fun_asr_nano" / "device_utils.py"
PACKAGE_MODEL_PATH = REPO_ROOT / "funasr" / "models" / "fun_asr_nano" / "model.py"
EXAMPLE_MODEL_PATH = (
    REPO_ROOT / "examples" / "industrial_data_pretraining" / "fun_asr_nano" / "model.py"
)


def _load_device_utils():
    spec = importlib.util.spec_from_file_location("fun_asr_nano_device_utils", DEVICE_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_autocast_device_type_keeps_npu_backend():
    device_utils = _load_device_utils()

    assert device_utils.resolve_autocast_device_type("npu:1") == "npu"
    assert device_utils.resolve_autocast_device_type("npu") == "npu"
    assert device_utils.resolve_autocast_device_type("cuda:0") == "cuda"
    assert device_utils.resolve_autocast_device_type("xpu:0") == "xpu"
    assert device_utils.resolve_autocast_device_type("mps") == "mps"
    assert device_utils.resolve_autocast_device_type("cpu") == "cpu"
    assert device_utils.resolve_autocast_device_type("unknown:0") == "cpu"


def test_resolve_autocast_device_type_accepts_device_like_objects():
    device_utils = _load_device_utils()

    class DeviceLike:
        type = "npu"

    assert device_utils.resolve_autocast_device_type(DeviceLike()) == "npu"


def test_fun_asr_nano_autocast_calls_use_shared_resolver():
    old_inline_fallback = 'device_type if device_type in ["cuda", "xpu", "mps"] else "cpu"'

    for path in (PACKAGE_MODEL_PATH, EXAMPLE_MODEL_PATH):
        source = path.read_text(encoding="utf-8")
        assert old_inline_fallback not in source
        assert source.count("resolve_autocast_device_type(") >= 2
