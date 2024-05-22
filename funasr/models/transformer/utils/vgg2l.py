"""VGG2L module definition for custom encoder."""

from typing import Tuple, Union

import torch


class VGG2L(torch.nn.Module):
    """VGG2L module for custom encoder.

    Args:
        idim: Input dimension.
        odim: Output dimension.
        pos_enc: Positional encoding class.

    """

    def __init__(self, idim: int, odim: int, pos_enc: torch.nn.Module = None):
        """Construct a VGG2L object."""
        super().__init__()

        self.vgg2l = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 2)),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
        )

        if pos_enc is not None:
            self.output = torch.nn.Sequential(
                torch.nn.Linear(128 * ((idim // 2) // 2), odim), pos_enc
            )
        else:
            self.output = torch.nn.Linear(128 * ((idim // 2) // 2), odim)

    def forward(self, feats: torch.Tensor, feats_mask: torch.Tensor) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        """Forward VGG2L bottleneck.

        Args:
            feats: Feature sequences. (B, F, D_feats)
            feats_mask: Mask of feature sequences. (B, 1, F)

        Returns:
            vgg_output: VGG output sequences.
                   (B, sub(F), D_out) or ((B, sub(F), D_out), (B, sub(F), D_att))
            vgg_mask: Mask of VGG output sequences. (B, 1, sub(F))

        """
        feats = feats.unsqueeze(1)
        vgg_output = self.vgg2l(feats)

        b, c, t, f = vgg_output.size()

        vgg_output = self.output(vgg_output.transpose(1, 2).contiguous().view(b, t, c * f))

        if feats_mask is not None:
            vgg_mask = self.create_new_mask(feats_mask)
        else:
            vgg_mask = feats_mask

        return vgg_output, vgg_mask

    def create_new_mask(self, feats_mask: torch.Tensor) -> torch.Tensor:
        """Create a subsampled mask of feature sequences.

        Args:
            feats_mask: Mask of feature sequences. (B, 1, F)

        Returns:
            vgg_mask: Mask of VGG2L output sequences. (B, 1, sub(F))

        """
        vgg1_t_len = feats_mask.size(2) - (feats_mask.size(2) % 3)
        vgg_mask = feats_mask[:, :, :vgg1_t_len][:, :, ::3]

        vgg2_t_len = vgg_mask.size(2) - (vgg_mask.size(2) % 2)
        vgg_mask = vgg_mask[:, :, :vgg2_t_len][:, :, ::2]

        return vgg_mask
