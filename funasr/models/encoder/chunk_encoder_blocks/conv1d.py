"""Conv1d block for Transducer encoder."""

from typing import Optional, Tuple, Union

import torch


class Conv1d(torch.nn.Module):
    """Conv1d module definition.

    Args:
        input_size: Input dimension.
        output_size: Output dimension.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dilation: Spacing between the kernel points.
        groups: Number of blocked connections from input channels to output channels.
        bias: Whether to add a learnable bias to the output.
        batch_norm: Whether to use batch normalization after convolution.
        relu: Whether to use a ReLU activation after convolution.
        causal: Whether to use causal convolution (set to True if streaming).
        dropout_rate: Dropout rate.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        dilation: Union[int, Tuple] = 1,
        groups: Union[int, Tuple] = 1,
        bias: bool = True,
        batch_norm: bool = False,
        relu: bool = True,
        causal: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """Construct a Conv1d object."""
        super().__init__()

        if causal:
            self.lorder = kernel_size - 1
            stride = 1
        else:
            self.lorder = 0
            stride = stride

        self.conv = torch.nn.Conv1d(
            input_size,
            output_size,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        if relu:
            self.relu_func = torch.nn.ReLU()

        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(output_size)

        self.out_pos = torch.nn.Linear(input_size, output_size)

        self.input_size = input_size
        self.output_size = output_size

        self.relu = relu
        self.batch_norm = batch_norm
        self.causal = causal

        self.kernel_size = kernel_size
        self.padding = dilation * (kernel_size - 1)
        self.stride = stride

        self.cache = None

    def reset_streaming_cache(self, left_context: int, device: torch.device) -> None:
        """Initialize/Reset Conv1d cache for streaming.

        Args:
            left_context: Number of left frames during chunk-by-chunk inference.
            device: Device to use for cache tensor.

        """
        self.cache = torch.zeros(
            (1, self.input_size, self.kernel_size - 1), device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input sequences.

        Args:
            x: Conv1d input sequences. (B, T, D_in)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_in)
            mask: Source mask. (B, T)
            chunk_mask: Chunk mask. (T_2, T_2)

        Returns:
            x: Conv1d output sequences. (B, sub(T), D_out)
            mask: Source mask. (B, T) or (B, sub(T))
            pos_enc: Positional embedding sequences.
                       (B, 2 * (T - 1), D_att) or (B, 2 * (sub(T) - 1), D_out)

        """
        x = x.transpose(1, 2)

        if self.lorder > 0:
            x = torch.nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
        else:
            mask = self.create_new_mask(mask)
            pos_enc = self.create_new_pos_enc(pos_enc)

        x = self.conv(x)

        if self.batch_norm:
            x = self.bn(x)

        x = self.dropout(x)

        if self.relu:
            x = self.relu_func(x)

        x = x.transpose(1, 2)

        return x, mask, self.out_pos(pos_enc)

    def chunk_forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor,
        left_context: int = 0,
        right_context: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode chunk of input sequence.

        Args:
            x: Conv1d input sequences. (B, T, D_in)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_in)
            mask: Source mask. (B, T)
            left_context: Number of frames in left context.
            right_context: Number of frames in right context.

        Returns:
            x: Conv1d output sequences. (B, T, D_out)
            pos_enc: Positional embedding sequences. (B, 2 * (T - 1), D_out)

        """
        x = torch.cat([self.cache, x.transpose(1, 2)], dim=2)

        if right_context > 0:
            self.cache = x[:, :, -(self.lorder + right_context) : -right_context]
        else:
            self.cache = x[:, :, -self.lorder :]

        x = self.conv(x)

        if self.batch_norm:
            x = self.bn(x)

        x = self.dropout(x)

        if self.relu:
            x = self.relu_func(x)

        x = x.transpose(1, 2)

        return x, self.out_pos(pos_enc)

    def create_new_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Create new mask for output sequences.

        Args:
            mask: Mask of input sequences. (B, T)

        Returns:
            mask: Mask of output sequences. (B, sub(T))

        """
        if self.padding != 0:
            mask = mask[:, : -self.padding]

        return mask[:, :: self.stride]

    def create_new_pos_enc(self, pos_enc: torch.Tensor) -> torch.Tensor:
        """Create new positional embedding vector.

        Args:
            pos_enc: Input sequences positional embedding.
                     (B, 2 * (T - 1), D_in)

        Returns:
            pos_enc: Output sequences positional embedding.
                     (B, 2 * (sub(T) - 1), D_in)

        """
        pos_enc_positive = pos_enc[:, : pos_enc.size(1) // 2 + 1, :]
        pos_enc_negative = pos_enc[:, pos_enc.size(1) // 2 :, :]

        if self.padding != 0:
            pos_enc_positive = pos_enc_positive[:, : -self.padding, :]
            pos_enc_negative = pos_enc_negative[:, : -self.padding, :]

        pos_enc_positive = pos_enc_positive[:, :: self.stride, :]
        pos_enc_negative = pos_enc_negative[:, :: self.stride, :]

        pos_enc = torch.cat([pos_enc_positive, pos_enc_negative[:, 1:, :]], dim=1)

        return pos_enc
