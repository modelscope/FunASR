import sys
import types

import pytest

from funasr.download.runtime_sdk_download_tool import main


def test_snapshot_download_failure_preserves_original_cause(monkeypatch, tmp_path):
    cause = RuntimeError("snapshot backend unavailable")
    snapshot_module = types.ModuleType("modelscope.hub.snapshot_download")

    def snapshot_download(*args, **kwargs):
        raise cause

    snapshot_module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", types.ModuleType("modelscope"))
    monkeypatch.setitem(
        sys.modules, "modelscope.hub", types.ModuleType("modelscope.hub")
    )
    monkeypatch.setitem(
        sys.modules, "modelscope.hub.snapshot_download", snapshot_module
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runtime_sdk_download_tool",
            "--model-name",
            "missing/model",
            "--export-dir",
            str(tmp_path),
            "--export",
            "False",
        ],
    )

    with pytest.raises(RuntimeError, match="missing/model") as caught:
        main()

    assert caught.value.__cause__ is cause
