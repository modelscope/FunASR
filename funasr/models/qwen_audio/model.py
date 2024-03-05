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
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from funasr.register import tables


@tables.register("model_classes", "Qwen-Audio")
@tables.register("model_classes", "QwenAudio")
@tables.register("model_classes", "QwenAudioWarp")
class QwenAudioWarp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        model_or_path = kwargs.get("model_path", "QwenAudio")
        model = AutoModelForCausalLM.from_pretrained(model_or_path, device_map="cpu",
                                                     trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=True)

        
        self.model = model
        self.tokenizer = tokenizer
        
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
    

        meta_data = {}
        # meta_data["batch_data_time"] = -1

        sp_prompt = "<|startoftranscription|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
        query = f"<audio>{data_in}</audio>{sp_prompt}"
        audio_info = self.tokenizer.process_audio(query)
        inputs = self.tokenizer(query, return_tensors='pt', audio_info=audio_info)
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs, audio_info=audio_info)
        response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False, audio_info=audio_info)
        print(response)

        results = []
        result_i = {"key": key[0], "text": result.text}
    
        results.append(result_i)
    
        return results, meta_data


@tables.register("model_classes", "Qwen-Audio-Chat")
@tables.register("model_classes", "QwenAudioChat")
@tables.register("model_classes", "QwenAudioChatWarp")
class QwenAudioChatWarp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        model_or_path = kwargs.get("model_path", "QwenAudio")
        bf16 = kwargs.get("bf16", False)
        fp16 = kwargs.get("fp16", False)
        model = AutoModelForCausalLM.from_pretrained(model_or_path,
                                                     device_map="cpu",
                                                     bf16=bf16,
                                                     fp16=fp16,
                                                     trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=True)
        
        self.model = model
        self.tokenizer = tokenizer
    
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
        
        
        meta_data = {}

        prompt = kwargs.get("prompt", "what does the person say?")
        history = kwargs.get("kwargs", None)
        if data_in is not None:
            # 1st dialogue turn
            query = self.tokenizer.from_list_format([
                {'audio': data_in},  # Either a local path or an url
                {'text': prompt},
            ])
        else:
            query = prompt
        response, history = self.model.chat(self.tokenizer, query=query, history=history)
        print(response)
        # The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".

        results = []
        result_i = {"key": key[0], "text": result.text}
        
        results.append(result_i)
        
        return results, meta_data
