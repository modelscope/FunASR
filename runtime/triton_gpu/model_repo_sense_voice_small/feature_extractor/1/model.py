#!/bin/bash
#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack
import torch
import numpy as np
import kaldifeat
import _kaldifeat
from typing import List
import json
import yaml
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union


class LFR(torch.nn.Module):
    """Batch LFR: https://github.com/Mddct/devil-asr/blob/main/patch/lfr.py"""

    def __init__(self, m: int = 7, n: int = 6) -> None:
        """
        Actually, this implements stacking frames and skipping frames.
        if m = 1 and n = 1, just return the origin features.
        if m = 1 and n > 1, it works like skipping.
        if m > 1 and n = 1, it works like stacking but only support right frames.
        if m > 1 and n > 1, it works like LFR.
        """
        super().__init__()

        self.m = m
        self.n = n

        self.left_padding_nums = math.ceil((self.m - 1) // 2)

    def forward(
        self, input_tensor: torch.Tensor, input_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, D = input_tensor.size()
        n_lfr = torch.ceil(input_lens / self.n)

        prepad_nums = input_lens + self.left_padding_nums

        right_padding_nums = torch.where(
            self.m >= (prepad_nums - self.n * (n_lfr - 1)),
            self.m - (prepad_nums - self.n * (n_lfr - 1)),
            0,
        )

        T_all = self.left_padding_nums + input_lens + right_padding_nums

        new_len = T_all // self.n

        T_all_max = T_all.max().int()

        tail_frames_index = (input_lens - 1).view(B, 1, 1).repeat(1, 1, D)  # [B,1,D]

        tail_frames = torch.gather(input_tensor, 1, tail_frames_index)
        tail_frames = tail_frames.repeat(1, right_padding_nums.max().int(), 1)
        head_frames = input_tensor[:, 0:1, :].repeat(1, self.left_padding_nums, 1)

        # stack
        input_tensor = torch.cat([head_frames, input_tensor, tail_frames], dim=1)

        index = (
            torch.arange(T_all_max, device=input_tensor.device, dtype=input_lens.dtype)
            .unsqueeze(0)
            .repeat(B, 1)
        )  # [B, T_all_max]
        index_mask = index < (self.left_padding_nums + input_lens).unsqueeze(1)  # [B, T_all_max]

        tail_index_mask = torch.logical_not(index >= (T_all.unsqueeze(1))) & index_mask
        tail = torch.ones(T_all_max, dtype=input_lens.dtype, device=input_tensor.device).unsqueeze(
            0
        ).repeat(B, 1) * (
            T_all_max - 1
        )  # [B, T_all_max]
        indices = torch.where(torch.logical_or(index_mask, tail_index_mask), index, tail)
        input_tensor = torch.gather(input_tensor, 1, indices.unsqueeze(2).repeat(1, 1, D))

        input_tensor = input_tensor.unfold(1, self.m, step=self.n).transpose(2, 3)

        return input_tensor.reshape(B, -1, D * self.m), new_len


class WavFrontend:
    """Conventional frontend structure for ASR."""

    def __init__(
        self,
        cmvn_file: str = None,
        fs: int = 16000,
        window: str = "hamming",
        n_mels: int = 80,
        frame_length: int = 25,
        frame_shift: int = 10,
        filter_length_min: int = -1,
        filter_length_max: float = -1,
        lfr_m: int = 7,
        lfr_n: int = 6,
        dither: float = 1.0,
    ) -> None:

        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.lfr = LFR(lfr_m, lfr_n)
        self.cmvn_file = cmvn_file
        self.dither = dither

        if self.cmvn_file:
            self.cmvn = self.load_cmvn()

    def apply_cmvn_batch(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply CMVN with mvn data
        """
        batch, frame, dim = inputs.shape
        means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
        vars = np.tile(self.cmvn[1:2, :dim], (frame, 1))

        means = torch.from_numpy(means).to(inputs.device)
        vars = torch.from_numpy(vars).to(inputs.device)

        inputs = (inputs + means) * vars
        return inputs

    def load_cmvn(
        self,
    ) -> np.ndarray:
        with open(self.cmvn_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        means_list = []
        vars_list = []
        for i in range(len(lines)):
            line_item = lines[i].split()
            if line_item[0] == "<AddShift>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    add_shift_line = line_item[3 : (len(line_item) - 1)]
                    means_list = list(add_shift_line)
                    continue
            elif line_item[0] == "<Rescale>":
                line_item = lines[i + 1].split()
                if line_item[0] == "<LearnRateCoef>":
                    rescale_line = line_item[3 : (len(line_item) - 1)]
                    vars_list = list(rescale_line)
                    continue

        means = np.array(means_list).astype(np.float64)
        vars = np.array(vars_list).astype(np.float64)
        cmvn = np.array([means, vars])
        return cmvn


class Fbank(torch.nn.Module):
    def __init__(self, opts):
        super(Fbank, self).__init__()
        self.fbank = kaldifeat.Fbank(opts)

    def forward(self, waves: List[torch.Tensor]):
        return self.fbank(waves)


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
        self.device = "cuda"

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "speech")
        # Convert Triton types to numpy types
        output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        if output0_dtype == np.float32:
            self.output0_dtype = torch.float32
        else:
            self.output0_dtype = torch.float16

        # Get OUTPUT1 configuration
        output1_config = pb_utils.get_output_config_by_name(model_config, "speech_lengths")
        # Convert Triton types to numpy types
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config["data_type"])

        params = self.model_config["parameters"]

        for li in params.items():
            key, value = li
            value = value["string_value"]
            if key == "config_path":
                with open(str(value), "rb") as f:
                    config = yaml.load(f, Loader=yaml.Loader)
            if key == "cmvn_path":
                cmvn_path = str(value)
        config["frontend_conf"]["cmvn_file"] = cmvn_path

        opts = kaldifeat.FbankOptions()
        opts.frame_opts.dither = 1.0  # TODO: 0.0 or 1.0
        opts.frame_opts.window_type = config["frontend_conf"]["window"]
        opts.mel_opts.num_bins = int(config["frontend_conf"]["n_mels"])
        opts.frame_opts.frame_shift_ms = float(config["frontend_conf"]["frame_shift"])
        opts.frame_opts.frame_length_ms = float(config["frontend_conf"]["frame_length"])
        opts.frame_opts.samp_freq = int(config["frontend_conf"]["fs"])
        opts.device = torch.device(self.device)
        self.opts = opts
        self.feature_extractor = Fbank(self.opts)
        self.feature_size = opts.mel_opts.num_bins

        self.frontend = WavFrontend(**config["frontend_conf"])

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        wavs = []
        for waveform in waveform_list:
            wav = torch.from_numpy(waveform).float().squeeze().to(self.device)
            wavs.append(wav)

        features = self.feature_extractor(wavs)
        features_len = [feature.shape[0] for feature in features]
        speech = torch.zeros(
            (len(features), max(features_len), self.opts.mel_opts.num_bins),
            dtype=self.output0_dtype,
            device=self.device,
        )
        for i, feature in enumerate(features):
            speech[i, : int(features_len[i])] = feature
        speech_lens = torch.tensor(features_len, dtype=torch.int64).to(self.device)

        feats, feats_len = self.frontend.lfr(speech, speech_lens)
        feats_len = feats_len.type(torch.int32)

        feats = self.frontend.apply_cmvn_batch(feats)
        feats = feats.type(self.output0_dtype)

        return feats, feats_len

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
        batch_count = []
        total_waves = []
        batch_len = []
        responses = []
        for request in requests:

            input0 = pb_utils.get_input_tensor_by_name(request, "wav")
            input1 = pb_utils.get_input_tensor_by_name(request, "wav_lens")

            cur_b_wav = input0.as_numpy() * (1 << 15)  # b x -1
            # remove paddings, however, encoder may can't batch requests since different lengths.
            # cur_b_wav = cur_b_wav[:, : int(input1.as_numpy()[0])]
            batch_count.append(cur_b_wav.shape[0])

            # convert the bx-1 numpy array into a 1x-1 list of arrays
            cur_b_wav_list = [np.expand_dims(cur_b_wav[i], 0) for i in range(cur_b_wav.shape[0])]
            total_waves.extend(cur_b_wav_list)

        features, feats_len = self.extract_feat(total_waves)

        i = 0
        for batch in batch_count:
            speech = features[i : i + batch]
            speech_lengths = feats_len[i : i + batch].unsqueeze(1)

            speech, speech_lengths = speech.cpu(), speech_lengths.cpu()

            out0 = pb_utils.Tensor.from_dlpack("speech", to_dlpack(speech))
            out1 = pb_utils.Tensor.from_dlpack("speech_lengths", to_dlpack(speech_lengths))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0, out1])
            responses.append(inference_response)
            i += batch

        return responses
