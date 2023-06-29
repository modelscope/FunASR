#!/usr/bin/env python3
#  2021, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers PostEncoder."""

from funasr.modules.nets_utils import make_pad_mask
from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from typing import Tuple

import copy
import logging
import torch

try:
    from transformers import AutoModel

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTransformersPostEncoder(AbsPostEncoder):
    """Hugging Face Transformers PostEncoder."""

    def __init__(
        self,
        input_size: int,
        model_name_or_path: str,
    ):
        """Initialize the module."""
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        model = AutoModel.from_pretrained(model_name_or_path)

        if hasattr(model, "encoder"):
            self.transformer = model.encoder
        else:
            self.transformer = model

        if hasattr(self.transformer, "embed_tokens"):
            del self.transformer.embed_tokens
        if hasattr(self.transformer, "wte"):
            del self.transformer.wte
        if hasattr(self.transformer, "word_embedding"):
            del self.transformer.word_embedding

        self.pretrained_params = copy.deepcopy(self.transformer.state_dict())

        if (
            self.transformer.config.is_encoder_decoder
            or self.transformer.config.model_type in ["xlnet", "t5"]
        ):
            self.use_inputs_embeds = True
            self.extend_attention_mask = False
        elif self.transformer.config.model_type == "gpt2":
            self.use_inputs_embeds = True
            self.extend_attention_mask = True
        else:
            self.use_inputs_embeds = False
            self.extend_attention_mask = True

        self.linear_in = torch.nn.Linear(
            input_size, self.transformer.config.hidden_size
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        input = self.linear_in(input)

        args = {"return_dict": True}

        mask = (~make_pad_mask(input_lengths)).to(input.device).float()

        if self.extend_attention_mask:
            args["attention_mask"] = _extend_attention_mask(mask)
        else:
            args["attention_mask"] = mask

        if self.use_inputs_embeds:
            args["inputs_embeds"] = input
        else:
            args["hidden_states"] = input

        if self.transformer.config.model_type == "mpnet":
            args["head_mask"] = [None for _ in self.transformer.layer]

        output = self.transformer(**args).last_hidden_state

        return output, input_lengths

    def reload_pretrained_parameters(self):
        self.transformer.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Transformers model parameters reloaded!")

    def output_size(self) -> int:
        """Get the output size."""
        return self.transformer.config.hidden_size


def _extend_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask[:, None, None, :]
    mask = (1.0 - mask) * -10000.0
    return mask
