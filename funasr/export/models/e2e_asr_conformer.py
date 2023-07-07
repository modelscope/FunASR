import logging
import torch
import torch.nn as nn

from funasr.export.utils.torch_function import MakePadMask
from funasr.export.utils.torch_function import sequence_mask
from funasr.models.encoder.conformer_encoder import ConformerEncoder
from funasr.export.models.encoder.conformer_encoder import ConformerEncoder as ConformerEncoder_export
from funasr.models.decoder.transformer_decoder import TransformerDecoder as TransformerDecoder_export


class Conformer(nn.Module):
    """
    export conformer into onnx format
    """

    def __init__(
            self,
            model,
            max_seq_len=512,
            feats_dim=560,
            output_size=2048,
            model_name='model',
            **kwargs,
    ):
        super().__init__()
        onnx = False
        if "onnx" in kwargs:
            onnx = kwargs["onnx"]
        if isinstance(model.encoder, ConformerEncoder):
            self.encoder = ConformerEncoder_export(model.encoder, onnx=onnx)
        elif isinstance(model.decoder, TransformerDecoder):
            self.decoder = TransformerDecoder_export(model.decoder, onnx=onnx)
        
        self.feats_dim = feats_dim
        self.output_size = output_size
        self.model_name = model_name

        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)
        
    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
    ):
        # a. To device
        batch = {"speech": speech, "speech_lengths": speech_lengths}
        # batch = to_device(batch, device=self.device)
    
        enc, enc_len = self.encoder(**batch)
        mask = self.make_pad_mask(enc_len)[:, None, :]

        # fill the decoder input
        enc_size = self.encoder.output_size
        pre_acoustic_embeds = torch.randn(1, 1, enc_size)
        cache_num = len(self.model.decoder)
        cache = [
            torch.zeros((1, self.decoder.size, self.decoder.self_attn.kernel_size))
            for _ in range(cache_num)
        ]

        decoder_out, olens = self.decoder(enc, enc_len, pre_acoustic_embeds, cache)
        decoder_out = torch.log_softmax(decoder_out, dim=-1)
        # sample_ids = decoder_out.argmax(dim=-1)

        return decoder_out, olens

    def get_dummy_inputs(self):
        speech = torch.randn(2, 30, self.feats_dim)
        speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
        return (speech, speech_lengths)

    def get_dummy_inputs_txt(self, txt_file: str = "/mnt/workspace/data_fbank/0207/12345.wav.fea.txt"):
        import numpy as np
        fbank = np.loadtxt(txt_file)
        fbank_lengths = np.array([fbank.shape[0], ], dtype=np.int32)
        speech = torch.from_numpy(fbank[None, :, :].astype(np.float32))
        speech_lengths = torch.from_numpy(fbank_lengths.astype(np.int32))
        return (speech, speech_lengths)

    def get_input_names(self):
        return ['speech', 'speech_lengths']

    def get_output_names(self):
        return ['logits', 'token_num']

    def get_dynamic_axes(self):
        return {
            'speech': {
                0: 'batch_size',
                1: 'feats_length'
            },
            'speech_lengths': {
                0: 'batch_size',
            },
            'logits': {
                0: 'batch_size',
                1: 'logits_length'
            },
        }
