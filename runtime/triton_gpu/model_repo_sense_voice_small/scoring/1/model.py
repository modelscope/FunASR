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

import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack

import json
import os
import yaml

import sentencepiece as spm


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

        self.init_tokenizer(self.model_config["parameters"])

    def init_tokenizer(self, parameters):
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "tokenizer_path":
                tokenizer_path = value
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.Load(tokenizer_path)

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

        total_seq = 0
        logits_list, batch_count = [], []

        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "ctc_logits")

            logits = from_dlpack(in_0.to_dlpack())
            logits_list.append(logits)

            total_seq += logits.shape[0]
            batch_count.append(logits.shape[0])

        logits_batch = torch.cat(logits_list, dim=0)
        yseq_batch = logits_batch.argmax(axis=-1)
        yseq_batch = torch.unique_consecutive(yseq_batch, dim=-1)

        yseq_batch = yseq_batch.tolist()

        # Remove blank_id and EOS tokens
        token_int_batch = [list(filter(lambda x: x not in (0, 2), yseq)) for yseq in yseq_batch]

        hyps = []
        for i, token_int in enumerate(token_int_batch):
            hyp = self.tokenizer.DecodeIds(token_int)
            hyps.append(hyp)

        responses = []
        i = 0
        for batch in batch_count:
            sents = np.array(hyps[i : i + batch])
            out0 = pb_utils.Tensor("OUTPUT0", sents.astype(self.out0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
            responses.append(inference_response)
            i += batch

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
