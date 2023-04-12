from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from funasr.layers.abs_normalize import AbsNormalize
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.torch_utils.device_funcs import force_gatherable
from funasr.train.abs_espnet_model import AbsESPnetModel
from funasr.utils.mask import make_pad_mask

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class BestRQPretrainModel(AbsESPnetModel):
    """BestRQ Pretrain model"""

    def __init__(
            self,
            frontend: Optional[AbsFrontend],
            specaug: Optional[AbsSpecAug],
            normalize: Optional[AbsNormalize],
            preencoder: Optional[AbsPreEncoder],
            encoder: AbsEncoder,
            input_dim: int = 512,
            embedding_dim: int = 512,
            num_embeddings: int = 8192,
            num_codebooks: int = 16,
            dropout_rate: float = 0.1,
            mask_prob: float = 0.01,
            mask_length: int = 10,
            min_masks: int = 2,
            layer_norm_epsilon=1e-5,
    ):
        assert check_argument_types()

        super().__init__()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.num_updates = 0

        #mask related
        assert mask_prob > 0.0
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.min_masks = min_masks

        self.input_dropout = torch.nn.Dropout(dropout_rate)

        # [embedding_dim, num_embeddings]
        random_embedding_weight = torch.empty(num_codebooks,
                                              embedding_dim,
                                              num_embeddings,
                                              requires_grad=False)

        self.embeddings = torch.nn.init.normal_(random_embedding_weight)

        random_projection_weight = torch.empty(input_dim,
                                               embedding_dim,
                                               requires_grad=False)
        self.projection = torch.nn.init.xavier_normal_(
            random_projection_weight)

        mask_emb_weight = torch.Tensor(input_dim)
        mask_emb_weight.requires_grad = True
        self.mask_emb = torch.nn.init.normal_(mask_emb_weight, mean=0, std=0.1)

        self.input_layer_norm = torch.nn.LayerNorm(input_dim,
                                                   layer_norm_epsilon)

        self.encoder = encoder
        self.encoder_top_n_out = torch.nn.parameter.Parameter(
            torch.Tensor(num_codebooks, self.encoder.output_size(),
                         num_embeddings))

    def _compute_loss(self, input: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor):
        input = input.transpose(1, 3)  # [B, num_embeddings, T' num_codebooks]
        entropy = torch.nn.functional.cross_entropy(
            input, target, reduction='none')  # [B, T', num_codebooks]
        # stop gradient for non mask area
        loss = entropy * mask.unsqueeze(2)
        return loss.sum() / (mask.sum() * loss.size(2))

    def _forward_encoder_blocks(self, xs: torch.Tensor, xs_masks: torch.Tensor,
                                pos_emb: torch.Tensor, mask_pad: torch.Tensor):

        masks = xs_masks
        for layer in self.encoder.encoders:
            xs, masks, _, _ = layer(xs, xs_masks, pos_emb, mask_pad)
        if self.encoder.normalize_before:
            xs = self.encoder.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def _nearest_embedding_idx(self, xs: torch.Tensor) -> torch.Tensor:
        xs = self.input_layer_norm(xs)
        xs = self.input_dropout(xs)
        xs = torch.matmul(xs, self.projection.to(xs.device))

        B, T, C = xs.size()
        flattened_input = xs.view(-1, C)
        embeddings = self.embeddings.to(
            xs.device)  # [num_codebooks, embedding_dim, num_embeddings]
        # [num_codebooks, B*T, num_embeddings]
        distance = (
                torch.sum(flattened_input ** 2, dim=1, keepdim=True).unsqueeze(0) +
                torch.sum(embeddings ** 2, dim=1, keepdim=True) -
                2 * torch.matmul(flattened_input.unsqueeze(0), embeddings))

        out = torch.argmin(distance, dim=-1)  # [num_codebooks, B*T]
        out = out.transpose(0, 1)  # [B*T, num_codebooks]
        return out.reshape(B, T, -1)  # [B, T, num_codebooks]

    def _apply_mask(self,
                    xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = compute_mask_indices(xs.size()[:-1],
                                     self.mask_prob,
                                     self.mask_length,
                                     self.min_masks,
                                     device=xs.device)
        masks_expand = masks.unsqueeze(-1)  # [B, T, 1]

        mask_emb = self.mask_emb.to(xs.device).view(1, 1, -1)
        xs = torch.where(masks_expand, mask_emb, xs)
        return xs, masks

    def _forward_subsampling(
            self, xs: torch.Tensor, xs_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        assert xs_lens.dim() == 1, xs_lens.shape

        # for data-paralel
        xs = xs[:, : xs_lens.max()]

        if self.specaug is not None and self.training:
            xs, xs_lens = self.specaug(xs, xs_lens)

        if self.normalize is not None:
            xs, xs_lens = self.normalize(xs, xs_lens)

        if self.preencoder is not None:
            xs, xs_lens = self.preencoder(xs, xs_lens)

        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)

        xs, pos_emb, masks = self.encoder.embed(xs, masks)
        return xs, pos_emb, masks

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # Check that batch_size is unified
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape)

        self.encoder.set_num_updates(self.num_updates)

        # 1 forward subsampling
        xs, pos_emb, masks = self._forward_subsampling(speech, speech_lengths)
        unmasked_xs = xs

        # 2 mask features
        # 2.0 apply mask
        masked_xs, masked_masks = self._apply_mask(xs)
        # 2.1 get nearest embedding
        target_ids = self._nearest_embedding_idx(unmasked_xs)
        # 3 forward xxx-formaer block
        out, out_mask = self._forward_encoder_blocks(masked_xs, masks, pos_emb,
                                                     masks)
        # 4 get logits
        out = out.unsqueeze(1)  # [B, 1, T', dim]
        top_n_out = self.encoder_top_n_out.unsqueeze(
            0)  # [num_codebooks, dim, num_embeddings]
        out = torch.matmul(out,
                           top_n_out)  # [B, num_codebooks, T', num_embeddings]

        # 5 compute loss
        loss = self._compute_loss(out, target_ids,
                                  out_mask.squeeze(1) * masked_masks)
        return {"loss": loss}

    def collect_feats(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def _extract_feats(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def get_num_updates(self):
        return self.num_updates
