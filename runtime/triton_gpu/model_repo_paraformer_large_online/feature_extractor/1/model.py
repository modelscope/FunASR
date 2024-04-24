# Created on 2024-01-01
# Author: GuAn Zhu

# Modified from NVIDIA(https://github.com/wenet-e2e/wenet/blob/main/runtime/gpu/
# model_repo_stateful/feature_extractor/1/model.py)

import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack
import torch
import kaldifeat
from typing import List
import json
import numpy as np
import yaml
from collections import OrderedDict


class LimitedDict(OrderedDict):
    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length

    def __setitem__(self, key, value):
        if len(self) >= self.max_length:
            self.popitem(last=False)
        super().__setitem__(key, value)


class Fbank(torch.nn.Module):
    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        return self.fbank(waves)


class Feat(object):
    def __init__(self, seqid, offset_ms, sample_rate, frame_stride, device="cpu"):
        self.seqid = seqid
        self.sample_rate = sample_rate
        self.wav = torch.tensor([], device=device)
        self.offset = int(offset_ms / 1000 * sample_rate)
        self.frames = None
        self.frame_stride = int(frame_stride)
        self.device = device
        self.lfr_m = 7

    def add_wavs(self, wav: torch.tensor):
        wav = wav.to(self.device)
        self.wav = torch.cat((self.wav, wav), axis=0)

    def get_seg_wav(self):
        seg = self.wav[:]
        self.wav = self.wav[-self.offset :]
        return seg

    def add_frames(self, frames: torch.tensor):
        """
        frames: seq_len x feat_sz
        """
        if self.frames is None:
            self.frames = torch.cat((frames[0, :].repeat((self.lfr_m - 1) // 2, 1), frames), axis=0)
        else:
            self.frames = torch.cat([self.frames, frames], axis=0)

    def get_frames(self, num_frames: int):
        seg = self.frames[0:num_frames]
        self.frames = self.frames[self.frame_stride :]
        return seg


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args["model_config"])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        if "GPU" in model_config["instance_group"][0]["kind"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "speech")
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        if self.output0_dtype == np.float32:
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16

        self.feature_size = output0_config["dims"][-1]
        self.decoding_window = output0_config["dims"][-2]

        params = self.model_config["parameters"]
        for li in params.items():
            key, value = li
            value = value["string_value"]
            if key == "config_path":
                with open(str(value), "rb") as f:
                    config = yaml.load(f, Loader=yaml.Loader)

        opts = kaldifeat.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.window_type = config["frontend_conf"]["window"]
        opts.mel_opts.num_bins = int(config["frontend_conf"]["n_mels"])
        opts.frame_opts.frame_shift_ms = float(config["frontend_conf"]["frame_shift"])
        opts.frame_opts.frame_length_ms = float(config["frontend_conf"]["frame_length"])
        opts.frame_opts.samp_freq = int(config["frontend_conf"]["fs"])
        opts.device = torch.device(self.device)
        self.opts = opts
        self.feature_extractor = Fbank(self.opts)

        self.seq_feat = LimitedDict(1024)
        chunk_size_s = float(params["chunk_size_s"]["string_value"])

        sample_rate = opts.frame_opts.samp_freq
        frame_shift_ms = opts.frame_opts.frame_shift_ms
        frame_length_ms = opts.frame_opts.frame_length_ms

        self.chunk_size = int(chunk_size_s * sample_rate)
        self.frame_stride = (chunk_size_s * 1000) // frame_shift_ms
        self.offset_ms = self.get_offset(frame_length_ms, frame_shift_ms)
        self.sample_rate = sample_rate

    def get_offset(self, frame_length_ms, frame_shift_ms):
        offset_ms = 0
        while offset_ms + frame_shift_ms < frame_length_ms:
            offset_ms += frame_shift_ms
        return offset_ms

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        total_waves = []
        responses = []
        batch_seqid = []
        end_seqid = {}
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
            wav = from_dlpack(input0.to_dlpack())[0]
            # input1 = pb_utils.get_input_tensor_by_name(request, "wav_lens")
            # wav_len = from_dlpack(input1.to_dlpack())[0]
            wav_len = len(wav)
            if wav_len < self.chunk_size:
                temp = torch.zeros(self.chunk_size, dtype=torch.float32, device=self.device)
                temp[0:wav_len] = wav[:]
                wav = temp

            in_start = pb_utils.get_input_tensor_by_name(request, "START")
            start = in_start.as_numpy()[0][0]
            in_ready = pb_utils.get_input_tensor_by_name(request, "READY")
            ready = in_ready.as_numpy()[0][0]
            in_corrid = pb_utils.get_input_tensor_by_name(request, "CORRID")
            corrid = in_corrid.as_numpy()[0][0]
            in_end = pb_utils.get_input_tensor_by_name(request, "END")
            end = in_end.as_numpy()[0][0]

            if start:
                self.seq_feat[corrid] = Feat(
                    corrid, self.offset_ms, self.sample_rate, self.frame_stride, self.device
                )
            if ready:
                self.seq_feat[corrid].add_wavs(wav)

            batch_seqid.append(corrid)
            if end:
                end_seqid[corrid] = 1

            wav = self.seq_feat[corrid].get_seg_wav() * 32768
            total_waves.append(wav)
        features = self.feature_extractor(total_waves)
        for corrid, frames in zip(batch_seqid, features):
            self.seq_feat[corrid].add_frames(frames)
            speech = self.seq_feat[corrid].get_frames(self.decoding_window)
            out_tensor0 = pb_utils.Tensor("speech", torch.unsqueeze(speech, 0).to("cpu").numpy())
            output_tensors = [out_tensor0]
            response = pb_utils.InferenceResponse(output_tensors=output_tensors)
            responses.append(response)
            if corrid in end_seqid:
                del self.seq_feat[corrid]
        return responses

    def finalize(self):
        print("Remove feature extractor!")
