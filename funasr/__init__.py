"""Initialize funasr package."""

import importlib
import os
import pkgutil
import traceback


dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "version.txt")
with open(version_file, "r") as f:
    __version__ = f.read().strip()


_IMPORT_ERRORS = {}
_IMPORT_ERROR_TRACEBACKS = {}
_IMPORT_DEBUG = os.environ.get("FUNASR_IMPORT_DEBUG") == "1"
_STRICT_IMPORT = os.environ.get("FUNASR_STRICT_IMPORT") == "1"
_LAZY_EXPORTS = {
    "AutoModel": ("funasr.auto.auto_model", "AutoModel"),
    "AutoFrontend": ("funasr.auto.auto_frontend", "AutoFrontend"),
}


def _record_import_error(name, error):
    """Internal: record import error.
    
        Args:
            name: TODO.
            error: TODO.
        """
    _IMPORT_ERRORS[name] = f"{error.__class__.__name__}: {error}"
    _IMPORT_ERROR_TRACEBACKS[name] = traceback.format_exc()
    if _IMPORT_DEBUG:
        print(f"Failed to import {name}: {_IMPORT_ERRORS[name]}")


def get_import_errors():
    """Get import errors."""
    return dict(_IMPORT_ERRORS)


def get_import_error_tracebacks():
    """Get import error tracebacks."""
    return dict(_IMPORT_ERROR_TRACEBACKS)


def import_submodules(package, recursive=True):
    """Import submodules.
    
        Args:
            package: TODO.
            recursive: TODO.
        """
    if isinstance(package, str):
        try:
            package = importlib.import_module(package)
        except Exception as e:
            _record_import_error(package, e)
            if _STRICT_IMPORT:
                raise
            return {}
    results = {}
    if not isinstance(package, str):
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                results[name] = importlib.import_module(name)
            except Exception as e:
                _record_import_error(name, e)
                if _STRICT_IMPORT:
                    raise
                continue
            if recursive and is_pkg:
                results.update(import_submodules(name))
    return results


import_submodules(__name__)

os.environ["HYDRA_FULL_ERROR"] = "1"


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise ModuleNotFoundError(
                "FunASR requires PyTorch before using AutoModel. Install a "
                "platform-appropriate torch build first, for example: "
                "`python -m pip install torch torchaudio --index-url "
                "https://download.pytorch.org/whl/cu126` for CUDA 12.6, "
                "or follow https://pytorch.org/get-started/locally/.",
                name="torch",
            ) from error
        raise

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = ["AutoModel", "AutoFrontend", "__version__", "get_import_errors", "get_import_error_tracebacks"]
