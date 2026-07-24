#!/usr/bin/env python3
"""Prepare and optionally upload current ModelScope model-card guidance.

The script keeps high-traffic ModelScope cards aligned with the FunASR GitHub
README and PyPI release guidance. By default it writes patched README files to a
local output directory. Use --upload with a ModelScope access token to publish
only those README files back to their model repositories.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_ROOT = Path(
    "/cpfs_speech/user/zhifu.gzf/.cache/funasr-ops/modelscope-card-audit-20260723/models"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/cpfs_speech/user/zhifu.gzf/.cache/funasr-ops/modelscope-card-sync-20260723"
)


@dataclass(frozen=True)
class CardPatch:
    repo_id: str
    source: Path
    path_in_repo: str
    replacements: tuple[tuple[str, str], ...]
    markers: tuple[str, ...]

    def output_path(self, output_root: Path) -> Path:
        return output_root / self.repo_id.replace("/", "__") / self.path_in_repo


def _patches(source_root: Path) -> tuple[CardPatch, ...]:
    nano_en_env_old = """# Environment Setup 🐍

```shell
git clone https://github.com/QwenAudio/Fun-ASR.git
cd Fun-ASR
pip install -r requirements.txt
```"""
    nano_en_env_new = """# Environment Setup 🐍

Use FunASR v1.3.26 or newer for the current ModelScope hub path, OpenAI-compatible server, and vLLM fallback fixes:

```shell
python -m pip install -U "funasr>=1.3.26" modelscope
```

For GPU inference, install the PyTorch / torchaudio CUDA wheels that match your driver from [pytorch.org](https://pytorch.org/get-started/locally/) and verify CUDA before using `device="cuda"`:

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
```

Optional demo repository setup:

```shell
git clone https://github.com/QwenAudio/Fun-ASR.git
cd Fun-ASR
pip install -r requirements.txt
```"""
    nano_zh_env_old = """# 环境安装 🐍

```shell
git clone https://github.com/QwenAudio/Fun-ASR.git
cd Fun-ASR
pip install -r requirements.txt
```"""
    nano_zh_env_new = """# 环境安装 🐍

推荐使用 FunASR v1.3.26 或更新版本，以获得当前 ModelScope hub 路径、OpenAI 兼容服务和 vLLM fallback 修复：

```shell
python -m pip install -U "funasr>=1.3.26" modelscope
```

GPU 推理前，请按 [PyTorch 官网](https://pytorch.org/get-started/locally/)安装与你驱动匹配的 PyTorch / torchaudio CUDA wheel，并先确认 CUDA 可见：

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
```

如需运行 demo 仓库，可再执行：

```shell
git clone https://github.com/QwenAudio/Fun-ASR.git
cd Fun-ASR
pip install -r requirements.txt
```"""
    sensevoice_env_old = """推理之前，请务必更新funasr与modelscope版本

```shell
pip install -U funasr modelscope
```"""
    sensevoice_env_new = """推理之前，请务必更新 funasr 与 modelscope 版本。推荐使用 FunASR v1.3.26 或更新版本，以获得当前命令行、OpenAI 兼容服务、字幕输出和 ModelScope hub 路径修复：

```shell
python -m pip install -U "funasr>=1.3.26" modelscope
```

GPU 推理前，请按 [PyTorch 官网](https://pytorch.org/get-started/locally/)安装与你驱动匹配的 PyTorch / torchaudio CUDA wheel，并先确认 CUDA 可见：

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
```"""
    return (
        CardPatch(
            repo_id="FunAudioLLM/Fun-ASR-Nano-2512",
            source=source_root / "FunAudioLLM--Fun-ASR-Nano-2512/snapshots/master/README.md",
            path_in_repo="README.md",
            replacements=(
                (nano_en_env_old, nano_en_env_new),
                (
                    "## Quick start via ModelScope\n```shell\npip install -U funasr modelscope\n```",
                    "## Quick start via ModelScope\n```shell\npython -m pip install -U \"funasr>=1.3.26\" modelscope\n```",
                ),
            ),
            markers=("funasr>=1.3.26", "torch.cuda.is_available()", "Quick start via ModelScope"),
        ),
        CardPatch(
            repo_id="FunAudioLLM/Fun-ASR-Nano-2512",
            source=source_root / "FunAudioLLM--Fun-ASR-Nano-2512/snapshots/master/README_zh.md",
            path_in_repo="README_zh.md",
            replacements=((nano_zh_env_old, nano_zh_env_new),),
            markers=("funasr>=1.3.26", "torch.cuda.is_available()", "环境安装"),
        ),
        CardPatch(
            repo_id="iic/SenseVoiceSmall",
            source=source_root / "iic--SenseVoiceSmall/snapshots/master/README.md",
            path_in_repo="README.md",
            replacements=(
                (sensevoice_env_old, sensevoice_env_new),
                (
                    "#安装ModelScope\npip install modelscope",
                    "# 安装 ModelScope 与 FunASR\npython -m pip install -U \"funasr>=1.3.26\" modelscope",
                ),
            ),
            markers=("funasr>=1.3.26", "torch.cuda.is_available()", "OpenAI 兼容"),
        ),
    )


def prepare_cards(source_root: Path, output_root: Path) -> list[CardPatch]:
    prepared: list[CardPatch] = []
    for patch in _patches(source_root):
        if not patch.source.exists():
            raise FileNotFoundError(f"missing source card: {patch.source}")
        text = patch.source.read_text()
        for old, new in patch.replacements:
            if old not in text:
                raise ValueError(f"expected text not found in {patch.source}: {old[:80]!r}")
            text = text.replace(old, new)
        missing = [marker for marker in patch.markers if marker not in text]
        if missing:
            raise ValueError(f"prepared {patch.repo_id}:{patch.path_in_repo} missing {missing}")
        destination = patch.output_path(output_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        prepared.append(patch)
    return prepared


def upload_cards(prepared: Iterable[CardPatch], output_root: Path, token: str) -> None:
    try:
        from modelscope.hub.api import HubApi
    except ImportError as exc:  # pragma: no cover - depends on ops environment
        raise SystemExit("Install modelscope before using --upload") from exc

    api = HubApi()
    for patch in prepared:
        local_path = patch.output_path(output_root)
        print(f"uploading {patch.repo_id}:{patch.path_in_repo}")
        api.upload_file(
            repo_id=patch.repo_id,
            path_or_fileobj=str(local_path),
            path_in_repo=patch.path_in_repo,
            repo_type="model",
            token=token,
            commit_message="Update FunASR quickstart guidance",
            commit_description=(
                "Sync model-card install guidance to funasr>=1.3.26 and add "
                "CUDA wheel verification notes."
            ),
            revision="master",
            disable_tqdm=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--upload", action="store_true", help="upload prepared README files to ModelScope")
    parser.add_argument("--token-file", type=Path, help="ModelScope access token file for --upload")
    parser.add_argument("--token", help="ModelScope access token for --upload")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared = prepare_cards(args.source_root, args.output_root)
    for patch in prepared:
        print(f"prepared {patch.repo_id}:{patch.path_in_repo} -> {patch.output_path(args.output_root)}")
    if args.upload:
        token = args.token or (args.token_file.read_text().strip() if args.token_file else "")
        if not token:
            raise SystemExit("--upload requires --token or --token-file")
        upload_cards(prepared, args.output_root, token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
