([简体中文](./installation_zh.md) | English)

# Install the Python SDK

> **Only need native Fun-ASR-Nano transcription?** Start with [Transformers 5.17.0](../transformers_native.md). It loads the separate `-hf` checkpoint without the FunASR toolkit. This page covers the `funasr.AutoModel` toolkit path; do not mix dependencies, parameters or output contracts.

Use this guide for `from funasr import AutoModel`. For a packaged C++ service, start with [Docker and runtime images](./docker.md). After installation, continue to the [SDK tutorial](../tutorial/README.md).

## 1. Create an isolated environment

The commands below use an already installed Python 3.11 interpreter. This is an environment example, not a promise that every model/backend supports every Python version. [setup.py](../../setup.py) declares `python_requires=">=3.7.0"`, but resolved dependencies and individual models can require newer Python. [pyproject.toml](../../pyproject.toml) defines the build backend, not a locked inference environment.

Linux/macOS:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

An existing Conda environment is also suitable; see the [Conda installation reference](https://docs.conda.io/en/latest/miniconda.html#windows-installers). On Apple Silicon, use an arm64 interpreter and matching wheels throughout; do not mix an x86_64 Conda environment with arm64 packages.

## 2. Install PyTorch, then choose one FunASR route

Install matching `torch` and `torchaudio` builds for your interpreter, OS, and accelerator using the [PyTorch installer](https://pytorch.org/get-started/locally/) or [version compatibility reference](https://pytorch.org/get-started/previous-versions/). FunASR's core package does not select these builds for you. Do not infer wheel compatibility solely from a locally installed CUDA toolkit.

### PyPI: use a released package

```sh
python -m pip install --upgrade funasr
```

This installs the package available from your configured index, not the current Git checkout. A model or feature documented in this checkout may not be in that release. For reproducible work, replace the unpinned install with the exact version you have validated. Package version for this checkout is recorded in [version.txt](../../funasr/version.txt); it is not evidence of index availability.

### Source: use this checkout and its examples

From the root of an existing FunASR checkout:

```sh
python -m pip install -e .
git rev-parse HEAD
```

For a new checkout, the [repository](https://github.com/modelscope/FunASR) can be cloned first:

```sh
git clone https://github.com/modelscope/FunASR.git
cd FunASR
python -m pip install -e .
```

Editable installation imports code from that directory; record its commit and local modifications. It is not a separate copy of the code. The repository's `examples/`, `runtime/`, and documentation are not all installed as Python package data by a PyPI install.

In this checkout, `modelscope` and `huggingface_hub` are core dependencies, not optional follow-up installs. Model-specific extras remain separate. The `knf` extra provides `kaldi-native-fbank` for supported paths without torchaudio, and `silero` provides Silero VAD; neither is needed for the standard first example. Check [setup.py](../../setup.py) and the chosen model's guide before adding extras. Do not install conflicting model stacks into one environment.

## 3. Verify the interpreter and imports

Run from the same activated environment you will use for inference:

```sh
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip --version
python -m pip check
python -c "import funasr, torch, torchaudio; from funasr import AutoModel; print('funasr:', funasr.__version__, funasr.__file__); print('torch:', torch.__version__, 'torchaudio:', torchaudio.__version__); print('CUDA available:', torch.cuda.is_available()); print('AutoModel import OK')"
```

These are dependency/import checks, not a model download or inference test. `funasr.__file__` should point to the intended installation; run outside unrelated checkouts when checking a PyPI environment. CPU is an explicit starting choice in the tutorial. A successful import or accelerator availability check does not validate an individual model on that device.

For a registration/import failure in this source checkout, inspect `funasr.get_import_errors()` or set `FUNASR_IMPORT_DEBUG=1` before starting Python. Missing optional model dependencies do not necessarily block your selected model. See [troubleshooting](../troubleshooting.md) before upgrading the entire environment.

## 4. Models, cache, and offline use

`AutoModel` accepts a model ID/alias or an existing local model directory. `hub="ms"` is the default; `hub="hf"` selects Hugging Face. Model files download separately from the Python package and use the hub client's cache. After loading, `model.model_path` identifies the resolved directory. Ensure enough writable disk space and preserve the complete directory, including configuration, tokenizer, frontend assets, and weights.

For repeatable or disconnected runs:

1. On an approved connected machine, acquire the exact model snapshot and dependencies. Record the full model ID, hub revision/commit, license, package versions, and file checksums. Also prepare every VAD, punctuation, or speaker model you plan to load.
2. Transfer complete model directories and dependency artifacts. Replace all hub aliases with those directories and all input URLs with local files. Check configuration references for other remote dependencies.
3. Use `disable_update=True` to skip FunASR's startup version check. This is **not** an offline switch for hub clients, model code, or missing assets. Validate the prepared environment with outbound network access disabled.

The following Python example requires a complete, reviewed Paraformer-compatible snapshot at `./models/paraformer-zh` and your local `./audio.wav`:

```python
from pathlib import Path
from funasr import AutoModel

model_dir = Path("./models/paraformer-zh").resolve()
audio = Path("./audio.wav").resolve()
assert model_dir.is_dir(), model_dir
assert audio.is_file(), audio
model = AutoModel(
    model=str(model_dir), device="cpu", disable_update=True,
    trust_remote_code=False,
)
print(model.generate(input=str(audio)))
```

Source caveat: the ModelScope helper passes `model_revision` to its downloader, but this checkout's Hugging Face helper calls `snapshot_download(model)` without forwarding the revision. Do not rely on `AutoModel(..., hub="hf", model_revision=...)` for pinning here; acquire a pinned snapshot with the hub client and pass its local directory. See the [download implementation](../../funasr/download/download_model_from_hub.py).

## 5. Trust and licensing

- Leave `trust_remote_code=False` unless a chosen model requires code you have reviewed. In the loader, enabling trust can install a model's `requirements.txt`; the ModelScope path can also import the configured `remote_code`. A local model directory is not automatically safe.
- Use isolated environments, reviewed model artifacts, and minimum filesystem/network permissions. Do not embed hub credentials in scripts, images, logs, or issue reports. Use the hub client's supported authentication mechanism.
- FunASR software is [MIT licensed](../../LICENSE). Model weights have separate terms; consult the exact model card and the [Model Zoo](../../model_zoo/readme.md). The [model license agreement](../../MODEL_LICENSE) applies only where the model card adopts it. Third-party models retain their own provenance and licenses.

Next: [produce your first transcript](../tutorial/README.md).
