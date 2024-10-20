import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rotary_embedding_torch import RotaryEmbedding
except:
    # print(
    #     "If you want use mossformer, lease install rotary_embedding_torch by: \n pip install -U rotary_embedding_torch"
    # )
    pass
from funasr.models.transformer.layer_norm import GlobalLayerNorm, CumulativeLayerNorm, ScaleNorm
from funasr.models.transformer.embedding import ScaledSinuEmbedding
from funasr.models.mossformer.mossformer import FLASH_ShareA_FFConvM


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type."""

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class MossformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        attn_dropout=0.1,
        norm_type="scalenorm",
        shift_tokens=True
    ):
        super().__init__()
        assert norm_type in (
            "scalenorm",
            "layernorm",
        ), "norm_type must be one of scalenorm or layernorm"

        if norm_type == "scalenorm":
            norm_klass = ScaleNorm
        elif norm_type == "layernorm":
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        self.layers = nn.ModuleList(
            [
                FLASH_ShareA_FFConvM(
                    dim=dim,
                    group_size=group_size,
                    query_key_dim=query_key_dim,
                    expansion_factor=expansion_factor,
                    causal=causal,
                    dropout=attn_dropout,
                    rotary_pos_emb=rotary_pos_emb,
                    norm_klass=norm_klass,
                    shift_tokens=shift_tokens,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, *, mask=None):
        ii = 0
        for flash in self.layers:
            x = flash(x, mask=mask)
            ii = ii + 1
        return x


class MossFormer_MaskNet(nn.Module):
    """The MossFormer module for computing output masks.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    num_blocks : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> mossformer_block = MossFormerM(1, 64, 8)
    >>> mossformer_masknet = MossFormer_MaskNet(64, 64, intra_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = mossformer_masknet(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=24,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super(MossFormer_MaskNet, self).__init__()
        self.num_spks = num_spks
        self.num_blocks = num_blocks
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.mdl = Computation_Block(
            num_blocks,
            out_channels,
            norm,
            skip_around_intra=skip_around_intra,
        )

        self.conv1d_out = nn.Conv1d(out_channels, out_channels * num_spks, kernel_size=1)
        self.conv1_decoder = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, S].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, S]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               S = the number of time frames
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d_encoder(x)
        if self.use_global_pos_enc:
            # x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
            #    x.size(1) ** 0.5)
            base = x
            x = x.transpose(1, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(0, -1)
            # print('base: {}, emb: {}'.format(base.shape, emb.shape))
            x = base + emb

        # [B, N, S]
        # for i in range(self.num_modules):
        #    x = self.dual_mdl[i](x)
        x = self.mdl(x)
        x = self.prelu(x)

        # [B, N*spks, S]
        x = self.conv1d_out(x)
        B, _, S = x.shape

        # [B*spks, N, S]
        x = x.view(B * self.num_spks, -1, S)

        # [B*spks, N, S]
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, S]
        x = self.conv1_decoder(x)

        # [B, spks, N, S]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, S]
        x = x.transpose(0, 1)

        return x


class MossFormerEncoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(MossFormerEncoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class MossFormerM(nn.Module):
    """This class implements the transformer encoder.

    Arguments
    ---------
    num_blocks : int
        Number of mossformer blocks to include.
    d_model : int
        The dimension of the input embedding.
    attn_dropout : float
        Dropout for the self-attention (Optional).
    group_size: int
        the chunk size
    query_key_dim: int
        the attention vector dimension
    expansion_factor: int
        the expansion factor for the linear projection in conv module
    causal: bool
        true for causal / false for non causal

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder_MossFormerM(num_blocks=8, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_blocks,
        d_model=None,
        causal=False,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        attn_dropout=0.1,
    ):
        super().__init__()

        self.mossformerM = MossformerBlock(
            dim=d_model,
            depth=num_blocks,
            group_size=group_size,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
            causal=causal,
            attn_dropout=attn_dropout,
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = self.mossformerM(src)
        output = self.norm(output)

        return output


class Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.

    Example
    ---------
        >>> comp_block = Computation_Block(64)
        >>> x = torch.randn(10, 64, 100)
        >>> x = comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100])
    """

    def __init__(
        self,
        num_blocks,
        out_channels,
        norm="ln",
        skip_around_intra=True,
    ):
        super(Computation_Block, self).__init__()

        ##MossFormer2M: MossFormer with recurrence
        # self.intra_mdl = MossFormer2M(num_blocks=num_blocks, d_model=out_channels)
        ##MossFormerM: the orignal MossFormer
        self.intra_mdl = MossFormerM(num_blocks=num_blocks, d_model=out_channels)
        self.skip_around_intra = skip_around_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, S].
            where, B = Batchsize,
               N = number of filters
               S = sequence time index
        """
        B, N, S = x.shape
        # intra RNN
        # [B, S, N]
        intra = x.permute(0, 2, 1).contiguous()  # .view(B, S, N)

        intra = self.intra_mdl(intra)

        # [B, N, S]
        intra = intra.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, S]
        if self.skip_around_intra:
            intra = intra + x

        out = intra
        return out
