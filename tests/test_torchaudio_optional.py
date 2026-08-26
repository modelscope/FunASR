"""Regression tests for the optional-torchaudio path.

FunASR can now run inference without torchaudio: fbank feature extraction
falls back to kaldi-native-fbank (``funasr.utils.fbank``) and audio decoding
falls back to soundfile/librosa. These tests exercise the torchaudio-absent
environment and assert that:

* the affected modules import cleanly when torchaudio is missing;
* torchaudio-only operations fail with an actionable ``ImportError`` instead
  of a bare ``AttributeError`` or a silently wrong result;
* the fbank fallback rejects unsupported torchaudio options and multi-channel
  waveforms explicitly instead of silently dropping them.
"""
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

# Run a snippet in a subprocess where importing torchaudio / kaldi_native_fbank
# raises ImportError, simulating a clean environment with neither backend.
_BLOCK_IMPORTS = (
    "import builtins\n"
    "_orig_import = builtins.__import__\n"
    "def _blocked(name, *args, **kwargs):\n"
    "    if name == 'torchaudio' or name.startswith('torchaudio.'):\n"
    "        raise ImportError('torchaudio is blocked for this test')\n"
    "    if name == 'kaldi_native_fbank' or name.startswith('kaldi_native_fbank.'):\n"
    "        raise ImportError('kaldi_native_fbank is blocked for this test')\n"
    "    return _orig_import(name, *args, **kwargs)\n"
    "builtins.__import__ = _blocked\n"
)


def _run_isolated(code):
    """Execute ``code`` in a subprocess with both backends blocked."""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_IMPORTS + code],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    return result


class TestFbankWithoutTorchaudio(unittest.TestCase):

    def test_fbank_module_imports_cleanly(self):
        # A clean environment (no torchaudio, no kaldi-native-fbank) must still
        # import the module; only actually calling fbank() needs a backend.
        result = _run_isolated(
            "import funasr.utils.fbank as fb\n"
            "print('HAS_TORCHAUDIO', fb._HAS_TORCHAUDIO)\n"
            "print('HAS_KNF', fb._knf is not None)\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("HAS_TORCHAUDIO False", result.stdout)
        self.assertIn("HAS_KNF False", result.stdout)

    def test_fbank_raises_actionable_error_without_backend(self):
        result = _run_isolated(
            "import torch\n"
            "import funasr.utils.fbank as fb\n"
            "try:\n"
            "    fb.fbank(torch.zeros(16000))\n"
            "    print('NO_ERROR')\n"
            "except ImportError as exc:\n"
            "    print('IMPORTERROR', 'kaldi-native-fbank' in str(exc), 'funasr[knf]' in str(exc))\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IMPORTERROR True True", result.stdout)

    def test_fbank_rejects_unsupported_options(self):
        import funasr.utils.fbank as fb

        fake_knf = mock.Mock()
        with mock.patch.object(fb, "_knf", fake_knf):
            # torchaudio-only options must be rejected explicitly, not dropped.
            with self.assertRaises(NotImplementedError):
                fb._fbank_knf(torch.zeros(16000), subtract_mean=True)
            with self.assertRaises(NotImplementedError):
                fb._fbank_knf(torch.zeros(16000), min_duration=0.5)
            with self.assertRaises(NotImplementedError):
                fb._fbank_knf(torch.zeros(16000), vtln_warp=0.9)

    def test_fbank_rejects_multichannel_without_channel(self):
        import funasr.utils.fbank as fb

        fake_knf = mock.Mock()
        fake_opts = mock.Mock()
        fake_fbank = mock.Mock()
        fake_fbank.num_frames_ready = 1
        fake_fbank.get_frame.return_value = np.zeros(23, dtype=np.float32)
        fake_knf.FbankOptions.return_value = fake_opts
        fake_knf.OnlineFbank.return_value = fake_fbank
        with mock.patch.object(fb, "_knf", fake_knf):
            # 2-D stereo waveform without `channel` is unsupported by the
            # single-channel kaldi-native-fbank backend.
            with self.assertRaises(NotImplementedError):
                fb._fbank_knf(torch.zeros(2, 16000))

    def test_fbank_honors_channel_selection(self):
        import funasr.utils.fbank as fb

        fake_knf = mock.Mock()
        fake_opts = mock.Mock()
        fake_fbank = mock.Mock()
        fake_fbank.num_frames_ready = 1
        fake_fbank.get_frame.return_value = np.zeros(23, dtype=np.float32)
        fake_knf.FbankOptions.return_value = fake_opts
        fake_knf.OnlineFbank.return_value = fake_fbank
        with mock.patch.object(fb, "_knf", fake_knf):
            # `channel=1` selects the second channel of a stereo waveform and
            # feeds a 1-D signal to the backend.
            fb._fbank_knf(torch.zeros(2, 16000), channel=1, num_mel_bins=23)
            accept_args = fake_fbank.accept_waveform.call_args[0]
            self.assertEqual(int(accept_args[0]), 16000)
            self.assertEqual(len(accept_args[1]), 16000)


class TestModulesImportWithoutTorchaudio(unittest.TestCase):

    def test_fun_asr_nano_utils_imports_cleanly(self):
        result = _run_isolated(
            "import funasr.models.fun_asr_nano.tools.utils as u\n"
            "print('OK')\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_fun_asr_nano_forced_align_propagates_dependency_error(self):
        result = _run_isolated(
            "import torch\n"
            "import funasr.models.fun_asr_nano.tools.utils as u\n"
            "try:\n"
            "    u.forced_align(torch.zeros(10, 100), torch.tensor([1, 2]))\n"
            "    print('NO_ERROR')\n"
            "except ImportError as exc:\n"
            "    print('IMPORTERROR', 'torchaudio' in str(exc))\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IMPORTERROR True", result.stdout)

    def test_paraformer_v2_imports_cleanly(self):
        result = _run_isolated(
            "import funasr.models.paraformer_v2_community.model\n"
            "print('OK')\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_audio_datasets_preprocessor_imports_cleanly(self):
        result = _run_isolated(
            "import funasr.datasets.audio_datasets.preprocessor\n"
            "print('OK')\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)


class TestTorchaudioCompat(unittest.TestCase):

    def test_require_torchaudio_raises_actionable_error(self):
        from funasr.utils.torchaudio_compat import require_torchaudio

        with mock.patch.dict(sys.modules, {"torchaudio": None}):
            with self.assertRaises(ImportError) as cm:
                require_torchaudio("forced alignment")
        message = str(cm.exception)
        self.assertIn("forced alignment", message)
        self.assertIn("torchaudio", message)


if __name__ == "__main__":
    unittest.main()