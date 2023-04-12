"""ConvInput block for Transducer encoder."""

from typing import Optional, Tuple, Union

import torch
import math

from funasr.modules.nets_utils import sub_factor_to_params, pad_to_len


class ConvInput(torch.nn.Module):
    """ConvInput module definition.

    Args:
        input_size: Input size.
        conv_size: Convolution size.
        subsampling_factor: Subsampling factor.
        vgg_like: Whether to use a VGG-like network.
        output_size: Block output dimension.

    """

    def __init__(
        self,
        input_size: int,
        conv_size: Union[int, Tuple],
        subsampling_factor: int = 4,
        vgg_like: bool = True,
        output_size: Optional[int] = None,
    ) -> None:
        """Construct a ConvInput object."""
        super().__init__()
        if vgg_like:
            if subsampling_factor == 1:
                conv_size1, conv_size2 = conv_size

                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(1, conv_size1, 3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(conv_size1, conv_size1, 3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d((1, 2)),
                    torch.nn.Conv2d(conv_size1, conv_size2, 3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(conv_size2, conv_size2, 3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d((1, 2)),
                )

                output_proj = conv_size2 * ((input_size // 2) // 2)
 
                self.subsampling_factor = 1

                self.stride_1 = 1

                self.create_new_mask = self.create_new_vgg_mask

            else:
                conv_size1, conv_size2 = conv_size

                kernel_1 = int(subsampling_factor / 2)

                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(1, conv_size1, 3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(conv_size1, conv_size1, 3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d((kernel_1, 2)),
                    torch.nn.Conv2d(conv_size1, conv_size2, 3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(conv_size2, conv_size2, 3, stride=1, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d((2, 2)),
                )

                output_proj = conv_size2 * ((input_size // 2) // 2)

                self.subsampling_factor = subsampling_factor

                self.create_new_mask = self.create_new_vgg_mask
                
                self.stride_1 = kernel_1

        else:
            if subsampling_factor == 1:
                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(1, conv_size, 3, [1,2], [1,0]),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(conv_size, conv_size, 3, [1,2], [1,0]),
                    torch.nn.ReLU(),
                )

                output_proj = conv_size * (((input_size - 1) // 2 - 1) // 2)

                self.subsampling_factor = subsampling_factor
                self.kernel_2 = 3
                self.stride_2 = 1

                self.create_new_mask = self.create_new_conv2d_mask

            else:
                kernel_2, stride_2, conv_2_output_size = sub_factor_to_params(
                    subsampling_factor,
                    input_size,
                )

                self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(1, conv_size, 3, 2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(conv_size, conv_size, kernel_2, stride_2),
                    torch.nn.ReLU(),
                )

                output_proj = conv_size * conv_2_output_size

                self.subsampling_factor = subsampling_factor
                self.kernel_2 = kernel_2
                self.stride_2 = stride_2

                self.create_new_mask = self.create_new_conv2d_mask

        self.vgg_like = vgg_like
        self.min_frame_length = 7

        if output_size is not None:
            self.output = torch.nn.Linear(output_proj, output_size)
            self.output_size = output_size
        else:
            self.output = None
            self.output_size = output_proj

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor], chunk_size: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequences.

        Args:
            x: ConvInput input sequences. (B, T, D_feats)
            mask: Mask of input sequences. (B, 1, T)

        Returns:
            x: ConvInput output sequences. (B, sub(T), D_out)
            mask: Mask of output sequences. (B, 1, sub(T))

        """
        if mask is not None:
            mask = self.create_new_mask(mask)
            olens = max(mask.eq(0).sum(1))

        b, t, f = x.size()
        x = x.unsqueeze(1) # (b. 1. t. f)

        if chunk_size is not None:
            max_input_length = int(
                chunk_size * self.subsampling_factor * (math.ceil(float(t) / (chunk_size * self.subsampling_factor) ))
            )
            x = map(lambda inputs: pad_to_len(inputs, max_input_length, 1), x)
            x = list(x)
            x = torch.stack(x, dim=0)
            N_chunks = max_input_length // ( chunk_size * self.subsampling_factor)
            x = x.view(b * N_chunks, 1, chunk_size * self.subsampling_factor, f)

        x = self.conv(x)

        _, c, _, f = x.size()
        if chunk_size is not None:
            x = x.transpose(1, 2).contiguous().view(b, -1, c * f)[:,:olens,:]
        else:
            x = x.transpose(1, 2).contiguous().view(b, -1, c * f)

        if self.output is not None:
            x = self.output(x)

        return x, mask[:,:olens][:,:x.size(1)]

    def create_new_vgg_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create a new mask for VGG output sequences.

        Args:
            mask: Mask of input sequences. (B, T)

        Returns:
            mask: Mask of output sequences. (B, sub(T))

        """
        if self.subsampling_factor > 1:
            vgg1_t_len = mask.size(1) - (mask.size(1) % (self.subsampling_factor // 2 ))
            mask = mask[:, :vgg1_t_len][:, ::self.subsampling_factor // 2]

            vgg2_t_len = mask.size(1) - (mask.size(1) % 2)
            mask = mask[:, :vgg2_t_len][:, ::2]
        else:
            mask = mask

        return mask

    def create_new_conv2d_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new conformer mask for Conv2d output sequences.

        Args:
            mask: Mask of input sequences. (B, T)

        Returns:
            mask: Mask of output sequences. (B, sub(T))

        """
        if self.subsampling_factor > 1:
            return mask[:, :-2:2][:, : -(self.kernel_2 - 1) : self.stride_2]
        else:
            return mask

    def get_size_before_subsampling(self, size: int) -> int:
        """Return the original size before subsampling for a given size.

        Args:
            size: Number of frames after subsampling.

        Returns:
            : Number of frames before subsampling.

        """
        return size * self.subsampling_factor
