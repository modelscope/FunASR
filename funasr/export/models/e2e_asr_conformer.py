import os
import logging
import torch
import torch.nn as nn

from funasr.export.utils.torch_function import MakePadMask
from funasr.export.utils.torch_function import sequence_mask
from funasr.models.encoder.conformer_encoder import ConformerEncoder
from funasr.models.decoder.transformer_decoder import TransformerDecoder
from funasr.export.models.encoder.conformer_encoder import ConformerEncoder as ConformerEncoder_export
from funasr.export.models.decoder.xformer_decoder import XformerDecoder as TransformerDecoder_export

class Conformer(nn.Module):
    """
    export conformer into onnx format
    """

    def __init__(
            self,
            model,
            max_seq_len=512,
            feats_dim=560,
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
        self.model_name = model_name

        if onnx:
            self.make_pad_mask = MakePadMask(max_seq_len, flip=False)
        else:
            self.make_pad_mask = sequence_mask(max_seq_len, flip=False)

    def _export_model(self, model, verbose, path):
        dummy_input = model.get_dummy_inputs()
        model_script = model
        model_path = os.path.join(path, f'{model.model_name}.onnx')
        if not os.path.exists(model_path):
            torch.onnx.export(
                model_script,
                dummy_input,
                model_path,
                verbose=verbose,
                opset_version=14,
                input_names=model.get_input_names(),
                output_names=model.get_output_names(),
                dynamic_axes=model.get_dynamic_axes()
            )

    def _export_encoder_onnx(self, verbose, path):
        model_encoder = self.encoder
        self._export_model(model_encoder, verbose, path)

    def _export_decoder_onnx(self, verbose, path):
        model_decoder = self.decoder
        self._export_model(model_decoder, verbose, path)

    def _export_onnx(self, verbose, path):
        self._export_encoder_onnx(verbose, path)
        self._export_decoder_onnx(verbose, path)