import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange


def identity(t, *args, **kwargs):
    """Identity.
    
        Args:
            t: TODO.
            *args: Variable positional arguments.
            **kwargs: Additional keyword arguments.
        """
    return t


def append_dims(x, num_dims):
    """Append dims.
    
        Args:
            x: TODO.
            num_dims: TODO.
        """
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))


def exists(val):
    """Exists.
    
        Args:
            val: TODO.
        """
    return val is not None


def default(val, d):
    """Default.
    
        Args:
            val: TODO.
            d: TODO.
        """
    return val if exists(val) else d


def padding_to_multiple_of(n, mult):
    """Padding to multiple of.
    
        Args:
            n: TODO.
            mult: TODO.
        """
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class Transpose(nn.Module):
    """Wrapper class of torch.transpose() for Sequential module."""

    def __init__(self, shape: tuple):
        """Initialize Transpose.
        
            Args:
                shape: TODO.
            """
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        """Forward pass for training.
        
            Args:
                x: TODO.
            """
        return x.transpose(*self.shape)


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        """Initialize DepthwiseConv1d.
        
            Args:
                in_channels: TODO.
                out_channels: TODO.
                kernel_size: Size/dimension parameter.
                stride: TODO.
                padding: TODO.
                bias: TODO.
            """
        super(DepthwiseConv1d, self).__init__()
        assert (
            out_channels % in_channels == 0
        ), "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs):
        """Forward pass for training.
        
            Args:
                inputs: TODO.
            """
        return self.conv(inputs)


class ConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.
    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 17,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        """Initialize ConvModule.
        
            Args:
                in_channels: TODO.
                kernel_size: Size/dimension parameter.
                expansion_factor: TODO.
                dropout_p: TODO.
            """
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            DepthwiseConv1d(
                in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2
            ),
        )

    def forward(self, inputs):
        """Forward pass for training.
        
            Args:
                inputs: TODO.
            """
        return inputs + self.sequential(inputs).transpose(1, 2)


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        """Initialize OffsetScale.
        
            Args:
                dim: TODO.
                heads: TODO.
            """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        """Forward pass for training.
        
            Args:
                x: TODO.
            """
        out = einsum("... d, h d -> ... h d", x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class FFConvM(nn.Module):
    def __init__(self, dim_in, dim_out, norm_klass=nn.LayerNorm, dropout=0.1):
        """Initialize FFConvM.
        
            Args:
                dim_in: TODO.
                dim_out: TODO.
                norm_klass: TODO.
                dropout: TODO.
            """
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
    ):
        """Forward pass for training.
        
            Args:
                x: TODO.
            """
        output = self.mdl(x)
        return output


class FLASH_ShareA_FFConvM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=1.0,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=True
    ):
        """Initialize FLASH_ShareA_FFConvM."""
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # positional embeddings
        self.rotary_pos_emb = rotary_pos_emb
        # norm
        self.dropout = nn.Dropout(dropout)
        # projections

        self.to_hidden = FFConvM(
            dim_in=dim,
            dim_out=hidden_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.to_qk = FFConvM(
            dim_in=dim,
            dim_out=query_key_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)

        self.to_out = FFConvM(
            dim_in=dim * 2,
            dim_out=dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

        self.gateActivate = nn.Sigmoid()

    def forward(self, x, *, mask=None):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        normed_x = x

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen
        residual = x

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.0)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        # initial projections

        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)

        # offset and scale
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u)
        out = (att_u * v) * self.gateActivate(att_v * u)
        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        """Cal attention.
        
            Args:
                x: TODO.
                quad_q: TODO.
                lin_q: TODO.
                quad_k: TODO.
                lin_k: TODO.
                v: TODO.
                u: TODO.
                mask: TODO.
            """
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        if exists(mask):
            lin_mask = rearrange(mask, "... -> ... 1")
            lin_k = lin_k.masked_fill(~lin_mask, 0.0)

        # rotate queries and keys

        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k)
            )

        # padding for groups

        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.0),
                (quad_q, quad_k, lin_q, lin_k, v, u),
            )

            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        # group along sequence

        quad_q, quad_k, lin_q, lin_k, v, u = map(
            lambda t: rearrange(t, "b (g n) d -> b g n d", n=self.group_size),
            (quad_q, quad_k, lin_q, lin_k, v, u),
        )

        if exists(mask):
            mask = rearrange(mask, "b (g j) -> b g 1 j", j=g)

        # calculate quadratic attention output

        sim = einsum("... i d, ... j d -> ... i j", quad_q, quad_k) / g

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.0)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.0)

        quad_out_v = einsum("... i j, ... j d -> ... i d", attn, v)
        quad_out_u = einsum("... i j, ... j d -> ... i d", attn, u)

        # calculate linear attention output

        if self.causal:
            lin_kv = einsum("b g n d, b g n e -> b g d e", lin_k, v) / g
            # exclusive cumulative sum along group dimension
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.0)
            lin_out_v = einsum("b g d e, b g n d -> b g n e", lin_kv, lin_q)

            lin_ku = einsum("b g n d, b g n e -> b g d e", lin_k, u) / g
            # exclusive cumulative sum along group dimension
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.0)
            lin_out_u = einsum("b g d e, b g n d -> b g n e", lin_ku, lin_q)
        else:
            lin_kv = einsum("b g n d, b g n e -> b d e", lin_k, v) / n
            lin_out_v = einsum("b g n d, b d e -> b g n e", lin_q, lin_kv)

            lin_ku = einsum("b g n d, b g n e -> b d e", lin_k, u) / n
            lin_out_u = einsum("b g n d, b d e -> b g n e", lin_q, lin_ku)

        # fold back groups into full sequence, and excise out padding
        return map(
            lambda t: rearrange(t, "b g n d -> b (g n) d")[:, :n],
            (quad_out_v + lin_out_v, quad_out_u + lin_out_u),
        )
