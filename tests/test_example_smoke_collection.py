"""Keep the standalone smoke CLIs collectable together with default pytest."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ("mcp_server", "openai_api")


@pytest.mark.parametrize("import_mode", ["prepend", "append", "importlib"])
def test_smoke_scripts_collect_together(tmp_path, import_mode):
    # Use real source files in the same directory layout, without local plugins.
    scripts = []
    for name in EXAMPLES:
        source = ROOT / "examples" / name
        target = tmp_path / "examples" / name
        target.mkdir(parents=True)
        for filename in ("smoke_test.py", "__init__.py"):
            if (source / filename).exists():
                shutil.copyfile(source / filename, target / filename)
        scripts.append(str(target / "smoke_test.py"))
    sentinel = tmp_path / "test_sentinel.py"
    sentinel.write_text("def test_collection_sentinel():\n    pass\n", encoding="utf-8")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "--import-mode=" + import_mode, *scripts, str(sentinel)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_sentinel.py::test_collection_sentinel" in result.stdout
    assert "1 test collected" in result.stdout


@pytest.mark.parametrize("example", EXAMPLES)
def test_smoke_script_help_still_runs_directly(example):
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / example / "smoke_test.py"), "--help"],
        cwd=ROOT / "examples" / example,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout
    assert "--timeout" in result.stdout
