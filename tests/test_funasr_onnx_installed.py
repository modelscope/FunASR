"""Run against an installed wheel/sdist, outside the source package directory."""

import ast
import builtins
import contextlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

if importlib.util.find_spec('funasr_onnx') is None:
    raise unittest.SkipTest('Install the standalone funasr-onnx distribution first.')

import funasr_onnx
from funasr_onnx.paraformer_online_bin import Paraformer as ParaformerOnline


ENTRYPOINTS = (
    funasr_onnx.Paraformer,
    funasr_onnx.ContextualParaformer,
    funasr_onnx.SeacoParaformer,
    ParaformerOnline,
    funasr_onnx.Fsmn_vad,
    funasr_onnx.Fsmn_vad_online,
    funasr_onnx.CT_Transformer,
    funasr_onnx.CT_Transformer_VadRealtime,
    funasr_onnx.SenseVoiceSmall,
)


class InstalledOnnxPackageTest(unittest.TestCase):
    def test_installed_distribution_version_and_origin(self):
        distribution = importlib.metadata.distribution('funasr-onnx')
        self.assertEqual(distribution.version, '0.4.3')
        self.assertEqual(
            Path(funasr_onnx.__file__).resolve().parent,
            Path(distribution.locate_file('funasr_onnx')).resolve(),
        )

    def test_installed_modules_have_no_string_exceptions(self):
        offenders = []
        for source in Path(funasr_onnx.__file__).parent.rglob('*.py'):
            for node in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
                if not isinstance(node, ast.Raise):
                    continue
                literal = node.exc
                if (
                    isinstance(literal, ast.Call)
                    and isinstance(literal.func, ast.Attribute)
                    and literal.func.attr == 'format'
                ):
                    literal = literal.func.value
                if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                    offenders.append(f'{source.name}:{node.lineno}')
        self.assertEqual(offenders, [])

    def check_export_import_error(self, error, expected_type, preserves_cause):
        original_import = builtins.__import__

        def broken_export_import(name, *args, **kwargs):
            if name == 'funasr':
                raise error
            return original_import(name, *args, **kwargs)

        # Only the optional exporter import is replaced; constructors are installed code.
        with tempfile.TemporaryDirectory() as model_dir:
            for entrypoint in ENTRYPOINTS:
                with self.subTest(entrypoint=entrypoint.__name__):
                    error.__traceback__ = None
                    with patch('builtins.__import__', side_effect=broken_export_import):
                        with contextlib.redirect_stdout(io.StringIO()):
                            with self.assertRaises(expected_type) as raised:
                                entrypoint(model_dir)
                    if preserves_cause:
                        self.assertIs(raised.exception.__cause__, error)
                    else:
                        self.assertIs(raised.exception, error)

    def test_missing_transitive_export_dependency_remains_visible(self):
        error = ModuleNotFoundError("No module named 'torchaudio'", name='torchaudio')
        self.check_export_import_error(error, ImportError, preserves_cause=True)

    def test_unrelated_export_import_failure_is_not_misreported(self):
        error = RuntimeError('exporter initialization failed')
        self.check_export_import_error(error, RuntimeError, preserves_cause=False)

    def test_failed_model_download_preserves_cause(self):
        original_import = builtins.__import__
        error = OSError('model download failed')

        def failed_download(*args, **kwargs):
            raise error

        def controlled_download_import(name, *args, **kwargs):
            if name == 'modelscope.hub.snapshot_download':
                return SimpleNamespace(snapshot_download=failed_download)
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            missing_model = str(Path(directory) / 'not-downloaded')
            for entrypoint in ENTRYPOINTS:
                with self.subTest(entrypoint=entrypoint.__name__):
                    error.__traceback__ = None
                    with patch('builtins.__import__', side_effect=controlled_download_import):
                        with self.assertRaises(RuntimeError) as raised:
                            entrypoint(missing_model)
                    self.assertIs(raised.exception.__cause__, error)

    def test_missing_onnxruntime_import_preserves_cause(self):
        script = '''
import builtins
import json
original_import = builtins.__import__
def broken_import(name, *args, **kwargs):
    if name == 'onnxruntime':
        raise ModuleNotFoundError("No module named 'onnxruntime'", name='onnxruntime')
    return original_import(name, *args, **kwargs)
builtins.__import__ = broken_import
try:
    import funasr_onnx
except Exception as error:
    print(json.dumps({
        'type': type(error).__name__,
        'cause': type(error.__cause__).__name__,
        'missing': getattr(error.__cause__, 'name', None),
    }))
else:
    print(json.dumps({'type': None}))
'''
        result = subprocess.run(
            [sys.executable, '-c', script], capture_output=True, text=True, timeout=60,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout.splitlines()[-1]), {
            'type': 'ImportError', 'cause': 'ModuleNotFoundError', 'missing': 'onnxruntime',
        })


if __name__ == '__main__':
    unittest.main()
