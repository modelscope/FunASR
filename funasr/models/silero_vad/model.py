"""Adapter that makes Silero VAD return FunASR-compatible millisecond segments."""

import time

import torch

from funasr.register import tables
from funasr.utils.load_utils import load_audio_text_image_video


@tables.register("model_classes", "SileroVad")
class SileroVad(torch.nn.Module):
    """Offline Silero VAD adapter used by ``AutoModel(vad_model='silero-vad')``.

    Requires the official ``silero-vad`` Python package.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as error:
            raise ImportError(
                "Silero VAD requires the optional dependency. Install it with "
                '`python -m pip install "funasr[silero]"` or '
                "`python -m pip install silero-vad`."
            ) from error
        self.onnx = bool(kwargs.get("silero_onnx", False))
        self.model = load_silero_vad(onnx=self.onnx)
        self.get_speech_timestamps = get_speech_timestamps

    @staticmethod
    def _split_long_segments(segments, max_single_segment_time):
        if not max_single_segment_time:
            return segments
        limit_ms = int(max_single_segment_time)
        if limit_ms <= 0:
            raise ValueError(
                "max_single_segment_time must resolve to a positive millisecond value"
            )
        split = []
        for start, end in segments:
            while end - start > limit_ms:
                split.append([start, start + limit_ms])
                start += limit_ms
            split.append([start, end])
        return split

    def inference(self, data_in, key=None, **kwargs):
        sample_rate = int(kwargs.get("silero_sampling_rate", 16000))
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports silero_sampling_rate=8000 or 16000")
        audio_list = load_audio_text_image_video(
            data_in,
            fs=sample_rate,
            audio_fs=kwargs.get("fs", sample_rate),
            data_type=kwargs.get("data_type", "sound"),
        )
        if not isinstance(audio_list, list):
            audio_list = [audio_list]

        started = time.perf_counter()
        results = []
        for index, audio in enumerate(audio_list):
            device = torch.device("cpu") if self.onnx else self.anchor.device
            waveform = torch.as_tensor(audio, dtype=torch.float32).flatten().to(device)
            timestamps = self.get_speech_timestamps(
                waveform,
                self.model,
                sampling_rate=sample_rate,
                threshold=kwargs.get("silero_threshold", 0.5),
                min_speech_duration_ms=kwargs.get("silero_min_speech_duration_ms", 250),
                min_silence_duration_ms=kwargs.get(
                    "silero_min_silence_duration_ms", 100
                ),
                speech_pad_ms=kwargs.get("silero_speech_pad_ms", 30),
            )
            segments = [
                [
                    int(item["start"] * 1000 / sample_rate),
                    int(item["end"] * 1000 / sample_rate),
                ]
                for item in timestamps
            ]
            segments = self._split_long_segments(
                segments, kwargs.get("max_single_segment_time")
            )
            results.append(
                {"key": key[index] if key else str(index), "value": segments}
            )
        elapsed = time.perf_counter() - started
        total_samples = sum(len(torch.as_tensor(audio)) for audio in audio_list)
        return results, {
            "batch_data_time": total_samples / sample_rate,
            "forward": elapsed,
        }
