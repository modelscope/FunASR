import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

import funasr
from funasr.auto.auto_model import AutoModel


class TestAutoModel(unittest.TestCase):

    def setUp(self):
        self.base_kwargs = {
            "model": "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
            "device": "cpu",
            "batch_size": 1,
            "disable_update": True,
        }

    def test_merge_thr_in_cb_model(self):
        kwargs = self.base_kwargs.copy()
        kwargs["spk_model"] = "cam++"
        merge_thr = 0.5
        kwargs["spk_kwargs"] = {"cb_kwargs": {"merge_thr": merge_thr}}
        model = AutoModel(**kwargs)
        self.assertEqual(model.cb_model.model_config['merge_thr'], merge_thr)
        # res = model.generate(input="/test.wav",
        #              batch_size_s=300)

    def test_progress_callback_called(self):
        class DummyModel:
            def __init__(self):
                self.param = torch.nn.Parameter(torch.zeros(1))

            def parameters(self):
                return iter([self.param])

            def eval(self):
                pass

            def inference(self, data_in=None, **kwargs):
                results = [{"text": str(d)} for d in data_in]
                return results, {"batch_data_time": 1}

        am = AutoModel.__new__(AutoModel)
        am.model = DummyModel()
        am.kwargs = {"batch_size": 2, "disable_pbar": True}

        progress = []

        res = AutoModel.inference(
            am,
            ["a", "b", "c"],
            progress_callback=lambda idx, total: progress.append((idx, total)),
        )

        self.assertEqual(len(progress), 2)
        self.assertEqual(progress, [(2, 3), (3, 3)])

    def test_import_submodules_records_failures(self):
        old_errors = dict(funasr._IMPORT_ERRORS)
        old_tracebacks = dict(funasr._IMPORT_ERROR_TRACEBACKS)
        package_name = "funasr_import_probe_pkg"

        with tempfile.TemporaryDirectory() as tmp_dir:
            package_dir = Path(tmp_dir) / package_name
            package_dir.mkdir()
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "broken.py").write_text(
                'raise RuntimeError("boom optional dependency")\n', encoding="utf-8"
            )
            sys.path.insert(0, tmp_dir)
            try:
                funasr._IMPORT_ERRORS.clear()
                funasr._IMPORT_ERROR_TRACEBACKS.clear()
                package = importlib.import_module(package_name)

                imported = funasr.import_submodules(package)

                failed_module = f"{package_name}.broken"
                self.assertEqual(imported, {})
                self.assertIn(failed_module, funasr._IMPORT_ERRORS)
                self.assertIn("boom optional dependency", funasr._IMPORT_ERRORS[failed_module])
                self.assertIn(failed_module, funasr._IMPORT_ERROR_TRACEBACKS)
            finally:
                sys.path.remove(tmp_dir)
                for module_name in list(sys.modules):
                    if module_name == package_name or module_name.startswith(package_name + "."):
                        sys.modules.pop(module_name, None)
                funasr._IMPORT_ERRORS.clear()
                funasr._IMPORT_ERRORS.update(old_errors)
                funasr._IMPORT_ERROR_TRACEBACKS.clear()
                funasr._IMPORT_ERROR_TRACEBACKS.update(old_tracebacks)

    def test_import_submodules_strict_import_raises(self):
        old_strict_import = funasr._STRICT_IMPORT
        package_name = "funasr_import_strict_probe_pkg"

        with tempfile.TemporaryDirectory() as tmp_dir:
            package_dir = Path(tmp_dir) / package_name
            package_dir.mkdir()
            (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (package_dir / "broken.py").write_text(
                'raise RuntimeError("strict optional dependency")\n', encoding="utf-8"
            )
            sys.path.insert(0, tmp_dir)
            try:
                funasr._STRICT_IMPORT = True
                package = importlib.import_module(package_name)

                with self.assertRaisesRegex(RuntimeError, "strict optional dependency"):
                    funasr.import_submodules(package)
            finally:
                funasr._STRICT_IMPORT = old_strict_import
                sys.path.remove(tmp_dir)
                for module_name in list(sys.modules):
                    if module_name == package_name or module_name.startswith(package_name + "."):
                        sys.modules.pop(module_name, None)

    def test_unregistered_model_reports_import_failures(self):
        old_errors = dict(funasr._IMPORT_ERRORS)
        try:
            funasr._IMPORT_ERRORS.clear()
            funasr._IMPORT_ERRORS["funasr.models.fake_model"] = (
                "ModuleNotFoundError: No module named 'fake_dep'"
            )

            with self.assertRaises(RuntimeError) as context:
                AutoModel.build_model(
                    model="missing_model",
                    model_conf={},
                    device="cpu",
                    ncpu=1,
                )

            message = str(context.exception)
            self.assertIn("missing_model", message)
            self.assertIn("funasr.models.fake_model", message)
            self.assertIn("fake_dep", message)
            self.assertIn("FUNASR_IMPORT_DEBUG", message)
            self.assertIn("FUNASR_STRICT_IMPORT", message)
        finally:
            funasr._IMPORT_ERRORS.clear()
            funasr._IMPORT_ERRORS.update(old_errors)


if __name__ == '__main__':
    unittest.main()
