"""Verify real distributions, not just a successful setuptools invocation."""

from email.parser import BytesParser
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime/python/libtorch"


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    directory = tmp_path_factory.mktemp("torch-dist")
    source = directory / "source"
    shutil.copytree(
        RUNTIME,
        source,
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", "dist"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(directory / "dist"),
            str(source),
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list((directory / "dist").glob("*.whl"))
    sdists = list((directory / "dist").glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1
    return wheels[0], sdists[0]


def test_wheel_contains_all_runtime_sources(distributions):
    wheel, _ = distributions
    expected = {
        str(path.relative_to(RUNTIME)): path.read_bytes()
        for path in (RUNTIME / "funasr_torch").rglob("*.py")
    }
    assert "funasr_torch/__init__.py" in expected
    assert "funasr_torch/utils/frontend.py" in expected
    with zipfile.ZipFile(wheel) as archive:
        assert archive.testzip() is None
        actual = {name for name in archive.namelist() if name.endswith(".py")}
        assert actual == set(expected)
        for name, content in expected.items():
            assert archive.read(name) == content


def test_sdist_contains_runtime_sources_and_build_files(distributions):
    _, sdist = distributions
    with tarfile.open(sdist, "r:gz") as archive:
        files = {
            str(Path(member.name).relative_to(Path(member.name).parts[0])): member
            for member in archive.getmembers()
            if member.isfile()
        }
        for path in (RUNTIME / "funasr_torch").rglob("*.py"):
            name = str(path.relative_to(RUNTIME))
            assert name in files, name
            assert archive.extractfile(files[name]).read() == path.read_bytes()
        for name in ["setup.py", "README.md"]:
            assert name in files
            assert (
                archive.extractfile(files[name]).read() == (RUNTIME / name).read_bytes()
            )


def metadata(wheel):
    with zipfile.ZipFile(wheel) as archive:
        paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(paths) == 1
        return BytesParser().parsebytes(archive.read(paths[0]))


@pytest.mark.parametrize(
    "dependency",
    [
        "torch",
        "sentencepiece",
        "librosa",
        "numpy",
        "scipy",
        "kaldi-native-fbank",
        "pyyaml",
    ],
)
def test_declares_direct_runtime_dependencies(distributions, dependency):
    wheel, _ = distributions
    names = {
        Requirement(item).name.lower().replace("_", "-")
        for item in metadata(wheel).get_all("Requires-Dist", [])
    }
    assert dependency in names


def test_torch_package_does_not_require_unrelated_runtime_engines(distributions):
    wheel, _ = distributions
    names = {
        Requirement(item).name.lower().replace("_", "-")
        for item in metadata(wheel).get_all("Requires-Dist", [])
    }
    assert not names & {"torch-quant", "onnxruntime", "onnxruntime-gpu"}

def test_distribution_version_excludes_empty_published_release(distributions):
    from packaging.version import Version

    wheel, sdist = distributions
    version = metadata(wheel)["Version"]
    assert Version(version) >= Version("0.1.4")
    with tarfile.open(sdist, "r:gz") as archive:
        member = next(item for item in archive.getmembers()
                      if item.name.count("/") == 1 and item.name.endswith("/PKG-INFO"))
        assert BytesParser().parsebytes(archive.extractfile(member).read())["Version"] == version


def test_wheel_includes_repository_license(distributions):
    wheel, _ = distributions
    with zipfile.ZipFile(wheel) as archive:
        licenses = [name for name in archive.namelist()
                    if ".dist-info/" in name and name.endswith("/LICENSE")]
        assert len(licenses) == 1
        assert archive.read(licenses[0]) == (ROOT / "LICENSE").read_bytes()
    assert metadata(wheel).get_all("License-File") == ["LICENSE"]


def test_sdist_includes_repository_license(distributions):
    _, sdist = distributions
    with tarfile.open(sdist, "r:gz") as archive:
        licenses = [item for item in archive.getmembers()
                    if item.name.count("/") == 1 and item.name.endswith("/LICENSE")]
        assert len(licenses) == 1
        assert archive.extractfile(licenses[0]).read() == (ROOT / "LICENSE").read_bytes()


def test_readme_source_install_directory_exists():
    lines = (RUNTIME / "README.md").read_text().splitlines()
    clone = lines.index("git clone https://github.com/modelscope/FunASR.git && cd FunASR")
    command = lines[clone + 1]
    assert command.startswith("cd ")
    directory = ROOT / command.removeprefix("cd ")
    assert directory.resolve() == RUNTIME.resolve()
    assert (directory / "setup.py").is_file()
