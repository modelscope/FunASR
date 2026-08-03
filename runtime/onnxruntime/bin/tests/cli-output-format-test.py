#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path


BINARIES = {
    "funasr-onnx-offline": ["--model-dir", "/missing/offline", "--wav-path", "/missing/audio.wav"],
    "funasr-onnx-offline-rtf": ["--model-dir", "/missing/offline", "--wav-path", "/missing/audio.wav"],
    "funasr-onnx-2pass": [
        "--model-dir",
        "/missing/offline",
        "--online-model-dir",
        "/missing/online",
        "--wav-path",
        "/missing/audio.wav",
    ],
    "funasr-onnx-2pass-rtf": [
        "--model-dir",
        "/missing/offline",
        "--online-model-dir",
        "/missing/online",
        "--wav-path",
        "/missing/audio.wav",
    ],
}


def run(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", type=Path, required=True)
    args = parser.parse_args()

    failures = []
    for name, required_args in BINARIES.items():
        binary = args.bin_dir / name
        if not binary.is_file():
            failures.append(f"{name}: executable not found at {binary}")
            continue

        help_result = run([str(binary), "--help"])
        help_output = help_result.stdout + help_result.stderr
        if help_result.returncode != 0:
            failures.append(f"{name}: --help exited {help_result.returncode}")
        if "--output-format" not in help_output:
            failures.append(f"{name}: --help does not expose --output-format")

        invalid_result = run(
            [str(binary), *required_args, "--output-format", "yaml"]
        )
        invalid_output = invalid_result.stdout + invalid_result.stderr
        if invalid_result.returncode != 2:
            failures.append(
                f"{name}: invalid format exited {invalid_result.returncode}, expected 2"
            )
        if "unsupported output format 'yaml'" not in invalid_output:
            failures.append(f"{name}: invalid format error is not actionable")
        if "FunASR init failed" in invalid_output:
            failures.append(f"{name}: invalid format reached model initialization")

    if failures:
        raise AssertionError("\n".join(failures))


if __name__ == "__main__":
    main()
