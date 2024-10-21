from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

import whisper

# import whisper_timestamped as whisper

from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

from funasr.register import tables


@tables.register("model_classes", "Whisper-tiny.en")
@tables.register("model_classes", "Whisper-tiny")
@tables.register("model_classes", "Whisper-base.en")
@tables.register("model_classes", "Whisper-base")
@tables.register("model_classes", "Whisper-small.en")
@tables.register("model_classes", "Whisper-small")
@tables.register("model_classes", "Whisper-medium.en")
@tables.register("model_classes", "Whisper-medium")
@tables.register("model_classes", "Whisper-large-v1")
@tables.register("model_classes", "Whisper-large-v2")
@tables.register("model_classes", "Whisper-large-v3")
@tables.register("model_classes", "Whisper-large-v3-turbo")
@tables.register("model_classes", "WhisperWarp")
class WhisperWarp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        hub = kwargs.get("hub", "funasr")
        if hub == "openai":
            model_or_path = kwargs.get("model_path", "Whisper-large-v3")
            if model_or_path.startswith("Whisper-"):
                model_or_path = model_or_path.replace("Whisper-", "")
            model = whisper.load_model(model_or_path)
        else:
            dims = kwargs.get("dims", {})
            dims = whisper.model.ModelDimensions(**dims)
            model = whisper.model.Whisper(dims=dims)

        self.model = model

        self.encoder_output_size = self.model.dims.n_audio_state

    def forward(
        self,
    ):
        pass

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        if kwargs.get("batch_size", 1) > 1:
            raise NotImplementedError("batch decoding is not implemented")

        if frontend is None and not hasattr(self, "frontend"):
            frontend_class = tables.frontend_classes.get("WhisperFrontend")
            frontend = frontend_class(
                n_mels=self.model.dims.n_mels, do_pad_trim=kwargs.get("do_pad_trim", True)
            )
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend

        meta_data = {}
        if (
            isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank"
        ):  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(
                data_in,
                fs=frontend.fs if hasattr(frontend, "fs") else 16000,
                audio_fs=kwargs.get("fs", 16000),
                data_type=kwargs.get("data_type", "sound"),
                tokenizer=tokenizer,
            )
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(
                audio_sample_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
            )
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
            lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])[0, :, :]
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        # # detect the spoken language
        # _, probs = self.model.detect_language(speech)
        # print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(**kwargs.get("DecodingOptions", {}))

        result = whisper.decode(self.model, speech, options=options)
        # result = whisper.transcribe(self.model, speech)

        results = []
        result_i = {"key": key[0], "text": result.text}

        results.append(result_i)

        return results, meta_data
