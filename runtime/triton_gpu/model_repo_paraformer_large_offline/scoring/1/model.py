#!/bin/bash
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack

import json
import os
import yaml


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

        # # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        # # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        self.init_vocab(self.model_config["parameters"])

    def init_vocab(self, parameters):
        blank_id = 0
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "blank_id":
                self.blank_id = int(value)
            elif key == "lm_path":
                lm_path = value
            elif key == "vocabulary":
                self.vocab_dict = self.load_vocab(value)
            if key == "ignore_id":
                ignore_id = int(value)

    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        with open(str(vocab_file), "rb") as f:
            vocab_list = json.load(f, encoding='utf-8')
        return vocab_list

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
        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        total_seq, max_token_num = 0, 0
        assert len(self.vocab_dict) == 8404, len(self.vocab_dict)
        logits_list, token_num_list = [], []

        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "logits")
            in_1 = pb_utils.get_input_tensor_by_name(request, "token_num")

            logits, token_num = from_dlpack(in_0.to_dlpack()), from_dlpack(in_1.to_dlpack()).cpu()
            max_token_num = max(max_token_num, token_num)

            assert logits.shape[0] == 1
            logits_list.append(logits)
            token_num_list.append(token_num)
            total_seq += 1

        logits_batch = torch.zeros(
            len(logits_list),
            max_token_num,
            len(self.vocab_dict),
            dtype=torch.float32,
            device=logits.device,
        )
        token_num_batch = torch.zeros(len(logits_list))

        for i, (logits, token_num) in enumerate(zip(logits_list, token_num_list)):
            logits_batch[i][: int(token_num)] = logits[0][: int(token_num)]
            token_num_batch[i] = token_num

        yseq_batch = logits_batch.argmax(axis=-1).tolist()
        token_int_batch = [list(filter(lambda x: x not in (0, 2), yseq)) for yseq in yseq_batch]

        tokens_batch = [[self.vocab_dict[i] for i in token_int] for token_int in token_int_batch]

        hyps = [
            "".join([t if t != "<space>" else " " for t in tokens]).encode("utf-8")
            for tokens in tokens_batch
        ]
        responses = []
        for i in range(total_seq):
            sents = np.array(hyps[i: i + 1])
            out0 = pb_utils.Tensor("OUTPUT0", sents.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
