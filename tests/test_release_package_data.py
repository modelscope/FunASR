from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGE_FILES = {
    "funasr/models/sense_voice/whisper_lib/normalizers/english.json",
    "funasr/models/rwkv_bat/cuda_encoder/wkv_cuda.cu",
    "funasr/models/rwkv_bat/cuda_encoder/wkv_op.cpp",
    "funasr/models/rwkv_bat/cuda_decoder/wkv_cuda.cu",
    "funasr/models/rwkv_bat/cuda_decoder/wkv_op.cpp",
}


def test_wheel_includes_runtime_package_data(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from setuptools import build_meta; "
            "build_meta.build_wheel(sys.argv[1])",
            str(tmp_path),
        ],
        check=True,
        cwd=ROOT,
    )
    [wheel_path] = tmp_path.glob("*.whl")

    with ZipFile(wheel_path) as wheel:
        wheel_files = set(wheel.namelist())

    assert RUNTIME_PACKAGE_FILES <= wheel_files
