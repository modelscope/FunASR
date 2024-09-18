import torch
from funasr.models.sense_voice.quantizer.quantization.vq import ResidualVectorQuantizer
import typing as tp


class CostumeQuantizer(torch.nn.Module):
    def __init__(
            self,
            input_size: int = 512,
            codebook_size: int = 1024,
            num_quantizers: int = 8,
            ema_decay: float = 0.95,
            kmeans_init: tp.Union[bool, str] = False,
            sampling_rate: int = 16_000,
            quantize_dropout: bool = False,
            rand_num_quant: tp.Optional[tp.List] = None,
            encoder_hop_length: int = 320,
            use_ddp: bool = True,
            q0_ds_ratio: int = 1,
            codec_dim: int = None,
            codec_range: float = None,
            threshold_ema_dead_code=2,
            **kwargs,
    ):
        super().__init__()
        if codec_dim is None:
            codec_dim = input_size

        self.input_proj, self.output_proj = None, None
        if codec_dim != input_size:
            self.input_proj = torch.nn.Linear(input_size, codec_dim)
            self.output_proj = torch.nn.Linear(codec_dim, input_size)

        self.input_act, self.codec_range = None, None
        if codec_range is not None:
            self.input_act = torch.nn.Tanh()
            self.codec_range = codec_range

        self.rq = ResidualVectorQuantizer(
            dimension=codec_dim,
            n_q=num_quantizers,
            bins=codebook_size,
            decay=ema_decay,
            kmeans_init=kmeans_init,
            quantize_dropout=quantize_dropout,
            rand_num_quant=rand_num_quant,
            encoder_hop_length=encoder_hop_length,
            use_ddp=use_ddp,
            q0_ds_ratio=q0_ds_ratio,
            threshold_ema_dead_code=threshold_ema_dead_code,
            **kwargs,
        )
        self.code_dim = codec_dim
        self.sampling_rate = sampling_rate
        self.bandwidth: tp.Optional[float] = None
        self.encoder_hop_length = encoder_hop_length
        self.codebook_size = codebook_size

    def forward(
            self,
            x,
            bandwidth: int = None,
    ):
        # x: input tensor in the shape of (B, T, C)
        # rq requires inputs in (B, C, T)

        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.input_act is not None:
            x = self.input_act(x) * self.codec_range

        qv = self.rq(x.permute(0, 2, 1), self.sampling_rate, bandwidth)
        x, indices, commit_loss, sub_quants = qv.quantized, qv.codes, qv.penalty, qv.sub_quants

        x = x.permute(0, 2, 1)
        if self.output_proj is not None:
            x = self.output_proj(x)

        return x, indices, commit_loss, sub_quants

    def inference(
            self,
            x,
            bandwidth: int = None,
    ):
        # x: input tensor in the shape of (B, T, C)
        # rq requires inputs in (B, C, T)
        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.input_act is not None:
            x = self.input_act(x) * self.codec_range

        qv = self.rq(x.permute(0, 2, 1), self.sampling_rate, bandwidth)
        x, indices, sub_quants = qv.quantized, qv.codes, qv.sub_quants

        x = x.permute(0, 2, 1)
        if self.output_proj is not None:
            x = self.output_proj(x)

        return x, indices, sub_quants

    def encode(
            self,
            x,
            bandwidth: int = None,
    ):
        # x: input tensor in the shape of (B, T, C)
        # rq requires inputs in (B, C, T)
        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.input_act is not None:
            x = self.input_act(x) * self.codec_range

        indices = self.rq.encode(x.permute(0, 2, 1), self.sampling_rate, bandwidth)
        # return value in n_q x B x T
        return indices

    def decode(self, indices):
        quantized_out = self.rq.decode(indices)
        # quantized_out in B x D x T
        if self.output_proj is not None:
            quantized_out = self.output_proj(quantized_out.transpose(1, 2)).transpose(1, 2)
        return quantized_out

    def output_size(self):
        return self.code_dim
