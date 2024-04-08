from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from . import whisper_lib as whisper
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

from funasr.register import tables


@tables.register("model_classes", "SenseVoice")
class SenseVoice(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        hub = kwargs.get("hub", "funasr")

        dims = kwargs.get("dims", {})
        dims = whisper.model.ModelDimensions(**dims)
        model = whisper.model.Whisper(dims=dims)
        
        self.model = model
        
        self.encoder_output_size = self.model.dims.n_audio_state
        
    def forward(self, ):
        pass
    
    def inference(self,
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
            frontend = frontend_class(n_mels=self.model.dims.n_mels, do_pad_trim=kwargs.get("do_pad_trim", True))
            self.frontend = frontend
        else:
            frontend = frontend if frontend is not None else self.frontend

        meta_data = {}
        if isinstance(data_in, torch.Tensor) and kwargs.get("data_type", "sound") == "fbank":  # fbank
            speech, speech_lengths = data_in, data_lengths
            if len(speech.shape) < 3:
                speech = speech[None, :, :]
            if speech_lengths is None:
                speech_lengths = speech.shape[1]
        else:
            # extract fbank feats
            time1 = time.perf_counter()
            audio_sample_list = load_audio_text_image_video(data_in, fs=frontend.fs if hasattr(frontend, "fs") else 16000, audio_fs=kwargs.get("fs", 16000),
                                                            data_type=kwargs.get("data_type", "sound"),
                                                            tokenizer=tokenizer)
            time2 = time.perf_counter()
            meta_data["load_data"] = f"{time2 - time1:0.3f}"
            speech, speech_lengths = extract_fbank(audio_sample_list, data_type=kwargs.get("data_type", "sound"),
                                                   frontend=frontend)
            time3 = time.perf_counter()
            meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
            frame_shift = frontend.frame_shift if hasattr(frontend, "frame_shift") else 10
            lfr_n = frontend.lfr_n if hasattr(frontend, "lfr_n") else 1
            meta_data["batch_data_time"] = speech_lengths.sum().item() * frame_shift * lfr_n / 1000

        speech = speech.to(device=kwargs["device"])[0, :, :]
        speech_lengths = speech_lengths.to(device=kwargs["device"])
        
        task = kwargs.get("task", "ASR")
        if isinstance(task, str):
            task = [task]
        task = "".join([f"<|{x}|>" for x in task])
        initial_prompt = kwargs.get("initial_prompt", f"<|startoftranscript|>{task}")
        language = kwargs.get("language", None)
        language = None if language == "auto" else language
        # if language is None:
        #     # detect the spoken language
        #     _, probs = self.model.detect_language(speech, initial_prompt=initial_prompt)
        #     print(f"Detected language: {max(probs, key=probs.get)}")
        #     language = max(probs, key=probs.get)
        #     language = language if kwargs.get("language", None) is None else kwargs.get("language")
        
        # decode the audio
        
        # initial_prompt = kwargs.get("initial_prompt", "<|startoftranscript|><|ASR|>")
        
        vocab_path = kwargs.get("vocab_path", None)
        options = whisper.DecodingOptions(language=language, fp16=False, without_timestamps=True, initial_prompt=initial_prompt, vocab_path=vocab_path)

        
        result = whisper.decode(self.model, speech, options)

        results = []
        result_i = {"key": key[0], "text": result.text}

        results.append(result_i)
    
        return results, meta_data
    