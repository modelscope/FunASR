import os
import subprocess
import sys
from pathlib import Path


def test_top_level_import_without_torch_succeeds():
    repo_root = Path(__file__).resolve().parents[1]
    script = r"""
import importlib.abc
import sys


class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return None


sys.meta_path.insert(0, BlockTorch())

import funasr

print(funasr.__version__)
assert isinstance(funasr.get_import_errors(), dict)
try:
    from funasr import AutoModel  # noqa: F401
except ModuleNotFoundError as error:
    assert error.name == "torch"
    assert "requires PyTorch before using AutoModel" in str(error)
else:
    raise AssertionError("AutoModel import should fail clearly when torch is missing")
"""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env.pop("FUNASR_STRICT_IMPORT", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
