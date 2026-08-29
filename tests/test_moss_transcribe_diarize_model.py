import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from funasr.models.moss_transcribe_diarize.model import (
    MossTranscribeDiarize,
    _parse_transcript,
    _result_from_transcript,
)


class MossTranscriptParserTest(unittest.TestCase):
    def test_parses_speaker_segments_and_preserves_numeric_brackets_in_text(self):
        transcript = "noise [0.48][S01]Welcome [2026][1.66]" "[12.26][S02]Ready[13.81]"

        self.assertEqual(
            _parse_transcript(transcript),
            [
                (0.48, 1.66, "S01", "Welcome [2026]"),
                (12.26, 13.81, "S02", "Ready"),
            ],
        )

    def test_normalizes_moss_output_to_funasr_result_contract(self):
        transcript = "[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]"

        result = _result_from_transcript("meeting", transcript)

        self.assertEqual(result["key"], "meeting")
        self.assertEqual(result["text"], "Welcome Ready")
        self.assertEqual(result["raw_text"], transcript)
        self.assertEqual(result["timestamp"], [[480, 1660], [12260, 13810]])
        self.assertEqual(
            result["sentence_info"],
            [
                {
                    "start": 480,
                    "end": 1660,
                    "text": "Welcome",
                    "sentence": "Welcome",
                    "spk": "S01",
                    "timestamp": [[480, 1660]],
                },
                {
                    "start": 12260,
                    "end": 13810,
                    "text": "Ready",
                    "sentence": "Ready",
                    "spk": "S02",
                    "timestamp": [[12260, 13810]],
                },
            ],
        )

    def test_keeps_unparsed_model_text_visible(self):
        result = _result_from_transcript("sample", "plain transcript")

        self.assertEqual(result["text"], "plain transcript")
        self.assertEqual(result["raw_text"], "plain transcript")
        self.assertEqual(result["timestamp"], [])
        self.assertEqual(result["sentence_info"], [])


class MossVllmBackendTest(unittest.TestCase):
    def test_auto_model_vllm_path_bypasses_weight_download(self):
        from funasr.auto.auto_model import AutoModel

        with patch(
            "funasr.auto.auto_model.download_model",
            side_effect=AssertionError(
                "vLLM client mode must not download model weights"
            ),
        ):
            auto_model = AutoModel(
                model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
                backend="vllm",
                vllm_base_url="http://vllm.test:8000/v1",
                device="cpu",
                disable_update=True,
            )

        self.assertIsInstance(auto_model.model, MossTranscribeDiarize)

    def test_rejects_external_vad_that_would_break_global_speaker_identity(self):
        with self.assertRaisesRegex(ValueError, "omit vad_model and spk_model"):
            MossTranscribeDiarize(
                backend="vllm",
                vllm_base_url="http://vllm.test:8000/v1",
                vad_model="fsmn-vad",
            )

    def test_posts_openai_compatible_transcription_request_without_loading_hf_model(
        self,
    ):
        session = MagicMock()
        response = MagicMock()
        response.json.return_value = {"text": "[0][S01]hello[1.5]"}
        session.post.return_value = response

        model = MossTranscribeDiarize(
            backend="vllm",
            vllm_base_url="http://vllm.test:8000/v1",
            vllm_model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
            vllm_api_key="secret",
            http_session=session,
        )
        results, metadata = model.inference(
            [np.zeros(1600, dtype=np.float32)],
            key=["sample"],
            prompt="transcribe and diarize",
            max_new_tokens=128,
        )

        self.assertEqual(results[0]["text"], "hello")
        self.assertEqual(results[0]["sentence_info"][0]["spk"], "S01")
        self.assertAlmostEqual(metadata["batch_data_time"], 0.1)
        request = session.post.call_args
        self.assertEqual(
            request.args[0],
            "http://vllm.test:8000/v1/audio/transcriptions",
        )
        self.assertEqual(
            request.kwargs["data"],
            {
                "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
                "response_format": "json",
                "temperature": "0",
                "prompt": "transcribe and diarize",
                "max_completion_tokens": "128",
            },
        )
        self.assertEqual(request.kwargs["headers"], {"Authorization": "Bearer secret"})
        filename, payload, content_type = request.kwargs["files"]["file"]
        self.assertEqual(filename, "audio.wav")
        self.assertIsInstance(payload, io.BytesIO)
        self.assertEqual(content_type, "audio/wav")
        response.raise_for_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
