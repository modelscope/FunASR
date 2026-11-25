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
import numpy as np
import soundfile as sf
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
    LOADERS = (funasr_onnx.Paraformer, ParaformerOnline,
               funasr_onnx.SenseVoiceSmall, funasr_onnx.Fsmn_vad)

    def test_long_and_empty_hotwords_stay_within_embedding_bounds(self):
        for cls in (funasr_onnx.ContextualParaformer, funasr_onnx.SeacoParaformer):
            with self.subTest(wrapper=cls.__name__):
                model = cls.__new__(cls)
                model.vocab = {'a': 3, 'b': 4}
                matrix, positions = model.proc_hotword(' ab  abababababab ')
                self.assertEqual(matrix.tolist(), [
                    [3, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                    [3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ])
                self.assertEqual(positions.tolist(), [1, 9, 0])
                for value in ('', '   \t\n'):
                    matrix, positions = model.proc_hotword(value)
                    self.assertEqual(matrix.tolist(), [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                    self.assertEqual(positions.tolist(), [0])

    def test_batched_timestamps_use_each_input_length(self):
        def features(lengths):
            return np.zeros((len(lengths), max(lengths), 4), dtype=np.float32), np.array(lengths)

        def acoustic_output(feats, lengths):
            peaks = np.zeros((len(lengths), feats.shape[1] * 3), dtype=np.float32)
            for row, length in enumerate(lengths):
                peaks[row, np.linspace(5, length * 3 - 6, 3).round().astype(int)] = 1
            return None, None, np.zeros_like(peaks), peaks

        # Acoustic outputs are synthetic. The installed batching, timestamp and
        # postprocessing code must keep the shorter sample independent of padding.
        for cls in (funasr_onnx.Paraformer, funasr_onnx.ContextualParaformer,
                    funasr_onnx.SeacoParaformer):
            with self.subTest(wrapper=cls.__name__):
                model = cls.__new__(cls)
                model.batch_size = 2
                model.plot_timestamp_to = ''
                model.language = 'zh'
                model.frontend = SimpleNamespace(opts=SimpleNamespace(
                    frame_opts=SimpleNamespace(samp_freq=16000)))
                model.load_data = lambda values, fs: values
                model.extract_feat = features
                model.decode = lambda scores, lens: [['\u4f60', '\u597d'] for _ in range(model._rows)]

                def infer(feats, lengths, *unused):
                    model._rows = len(lengths)
                    return acoustic_output(feats, lengths)

                model.infer = infer
                model.bb_infer = infer
                model.vocab = {'a': 3}
                model.eb_infer = lambda hotwords, lengths: [np.zeros((10, len(hotwords), 4), dtype=np.float32)]
                args = () if cls is funasr_onnx.Paraformer else ('a',)
                single = [model([length], *args)[0] for length in [17, 50]]
                self.assertEqual([x['timestamp'] for x in single], [
                    [[70, 470], [470, 945]], [[70, 670], [1450, 2925]],
                ])
                self.assertEqual(model([17, 50], *args), single)

    def test_terminal_abbreviation_preserves_text_and_timestamps(self):
        from funasr_onnx.utils.postprocess_utils import abbr_dispose

        self.assertEqual(abbr_dispose(['a', ' ', 'b']), ['AB'])
        self.assertEqual(abbr_dispose(['a', ' ', 'b'], [[0, 100], [100, 200]]),
                         (['AB'], [[0, 200]]))
        self.assertEqual(abbr_dispose(['a', ' ', 'b', ' ', 'c']), ['ABC'])

    def test_lossless_non_wav_without_sidecar(self):
        samples = (np.arange(4800, dtype=np.int16) - 2400).astype(np.float32) / 32768
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.flac'
            sf.write(path, samples, 16000, subtype='PCM_16')
            before = path.read_bytes()
            for cls in self.LOADERS:
                with self.subTest(loader=cls.__module__):
                    result = cls.__new__(cls).load_data(str(path), fs=16000)
                    self.assertEqual(len(result), 1)
                    np.testing.assert_array_equal(result[0], samples)
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ['sample.flac'])

    def test_wav_and_flac_list_preserves_input_order(self):
        first = np.full(800, 0.125, dtype=np.float32)
        second = np.full(1200, -0.25, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / 'first.wav', Path(directory) / 'second.flac']
            for path, samples in zip(paths, [first, second]):
                sf.write(path, samples, 16000, subtype='PCM_16')
            for cls in self.LOADERS:
                with self.subTest(loader=cls.__module__):
                    result = cls.__new__(cls).load_data([str(p) for p in paths], fs=16000)
                    self.assertEqual(len(result), 2)
                    np.testing.assert_array_equal(result[0], first)
                    np.testing.assert_array_equal(result[1], second)
                    self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['first.wav', 'second.flac'])

    def test_waveform_array_is_preserved(self):
        samples = np.linspace(-0.25, 0.25, 320, dtype=np.float32)
        for cls in self.LOADERS:
            with self.subTest(loader=cls.__module__):
                result = cls.__new__(cls).load_data(samples, fs=16000)
                self.assertEqual(len(result), 1)
                self.assertIs(result[0], samples)

    def test_invalid_audio_remains_visible_without_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.flac'
            path.write_bytes(b'not audio')
            for cls in self.LOADERS:
                with self.subTest(loader=cls.__module__):
                    with self.assertRaises(Exception):
                        cls.__new__(cls).load_data(str(path), fs=16000)
                    self.assertEqual(path.read_bytes(), b'not audio')
                    self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ['invalid.flac'])

    def test_installed_distribution_version_and_origin(self):
        distribution = importlib.metadata.distribution('funasr-onnx')
        self.assertEqual(distribution.version, '0.4.11')
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

