import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "examples"
    / "industrial_data_pretraining"
    / "qwen3_asr"
    / "transcribe_vllm_offline.py"
)


def load_example():
    spec = importlib.util.spec_from_file_location("qwen3_vllm_offline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResult:
    def __init__(self, text, language="Chinese"):
        self.text = text
        self.language = language


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, *, audio, language):
        self.calls.append((audio, language))
        return [FakeResult(f"chunk-{len(self.calls)}")]


class Qwen3AsrVllmOfflineExampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = load_example()

    def test_transcribe_chunks_preserves_offsets_and_language(self):
        self.assertTrue(hasattr(self.example, "transcribe_chunks"))
        model = FakeModel()
        chunks = [([0.1] * 16000, 0.0), ([0.2] * 8000, 3.25)]

        segments = self.example.transcribe_chunks(
            model, chunks, sample_rate=16000, language="Chinese"
        )

        self.assertEqual(
            segments,
            [
                {
                    "start_ms": 0,
                    "end_ms": 1000,
                    "text": "chunk-1",
                    "language": "Chinese",
                },
                {
                    "start_ms": 3250,
                    "end_ms": 3750,
                    "text": "chunk-2",
                    "language": "Chinese",
                },
            ],
        )
        self.assertTrue(all(language == "Chinese" for _, language in model.calls))

    def test_transcribe_chunks_rejects_result_count_mismatch(self):
        class EmptyModel:
            def transcribe(self, **kwargs):
                return []

        with self.assertRaisesRegex(RuntimeError, "one result"):
            self.example.transcribe_chunks(
                EmptyModel(), [([0.1], 0.0)], sample_rate=16000, language=None
            )

    def test_help_does_not_require_gpu_dependencies(self):
        self.assertTrue(hasattr(self.example, "build_parser"))
        parser = self.example.build_parser()
        args = parser.parse_args(["input.mp3"])

        self.assertEqual(args.audio, Path("input.mp3"))
        self.assertEqual(args.chunk_seconds, 180.0)
        self.assertEqual(args.max_inference_batch_size, 4)


if __name__ == "__main__":
    unittest.main()
