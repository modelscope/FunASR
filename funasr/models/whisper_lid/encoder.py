# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import copy
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import whisper

from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.specaug.specaug import SpecAug
from funasr.register import tables


@tables.register("encoder_classes", "OpenAIWhisperEncoderWarp")
class OpenAIWhisperEncoderWarp(nn.Module):
    """Transformer-based Speech Encoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    def __init__(
        self,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: str = None,
        use_specaug: bool = False,
        use_padmask: bool = False,
        specaug_conf: Union[dict, None] = None,
    ):
        super().__init__()

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(whisper_model, download_root=download_dir, device="cpu")
        self.encoders = copy.deepcopy(_model.encoder)
        self.encoders.train()

        del _model

        if use_specaug:
            self.specaug = SpecAug(**specaug_conf)
        else:
            self.specaug = None
        self.use_padmask = use_padmask

    def whisper_encode(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        x = F.gelu(self.encoders.conv1(input))
        x = F.gelu(self.encoders.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = self.encoders.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            # due to positional encoding, audios >30 sec won't be accepted
            x = x[:, :max_pos, :] + self.encoders.positional_embedding

        if ilens is not None:
            olens = (
                1
                + (ilens - self.encoders.conv2.kernel_size[0] + 2 * self.encoders.conv2.padding[0])
                // self.encoders.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        if self.use_padmask:
            padding_mask = (~make_pad_mask(olens)[:, None, :]).to(x.device)
        else:
            padding_mask = None

        x = self.dropout(x)

        for layer, block in enumerate(self.encoders.blocks):
            x = block(x)
            if layer < len(self.encoders.blocks) - 1:
                x = self.dropout(x)

        x = self.encoders.ln_post(x)

        return x, olens

    def output_size(self) -> int:
        # dummy output size
        return self.encoders.conv2.weight.shape[0]

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        feats, feats_lens = xs_pad, ilens

        if self.specaug is not None and self.encoders.training:
            feats = torch.transpose(feats, 1, 2)
            feats, feats_lens = self.specaug(feats, feats_lens)
            feats = torch.transpose(feats, 1, 2)

        xs_pad, olens = self.whisper_encode(feats, feats_lens)

        return xs_pad, olens, None
