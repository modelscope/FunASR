# from .ctc import CTC
# from .joint_network import JointNetwork
#
# # encoder
# from espnet2.asr.encoder.rnn_encoder import RNNEncoder as espnetRNNEncoder
# from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder as espnetVGGRNNEncoder
# from espnet2.asr.encoder.contextual_block_transformer_encoder import ContextualBlockTransformerEncoder as espnetContextualTransformer
# from espnet2.asr.encoder.contextual_block_conformer_encoder import ContextualBlockConformerEncoder as espnetContextualConformer
# from espnet2.asr.encoder.transformer_encoder import TransformerEncoder as espnetTransformerEncoder
# from espnet2.asr.encoder.conformer_encoder import ConformerEncoder as espnetConformerEncoder
# from funasr.export.models.encoder.rnn import RNNEncoder
# from funasr.export.models.encoders import TransformerEncoder
# from funasr.export.models.encoders import ConformerEncoder
# from funasr.export.models.encoder.contextual_block_xformer import ContextualBlockXformerEncoder
#
# # decoder
# from espnet2.asr.decoder.rnn_decoder import RNNDecoder as espnetRNNDecoder
# from espnet2.asr.transducer.transducer_decoder import TransducerDecoder as espnetTransducerDecoder
# from funasr.export.models.decoder.rnn import (
#     RNNDecoder
# )
# from funasr.export.models.decoders import XformerDecoder
# from funasr.export.models.decoders import TransducerDecoder
#
# # lm
# from espnet2.lm.seq_rnn_lm import SequentialRNNLM as espnetSequentialRNNLM
# from espnet2.lm.transformer_lm import TransformerLM as espnetTransformerLM
# from .language_models.seq_rnn import SequentialRNNLM
# from .language_models.transformer import TransformerLM
#
# # frontend
# from espnet2.asr.frontend.s3prl import S3prlFrontend as espnetS3PRLModel
# from .frontends.s3prl import S3PRLModel
#
# from espnet2.asr.encoder.sanm_encoder import SANMEncoder_tf, SANMEncoderChunkOpt_tf
# from espnet_onnx.export.asr.models.encoders.transformer_sanm import TransformerEncoderSANM_tf
# from espnet2.asr.decoder.transformer_decoder import FsmnDecoderSCAMAOpt_tf
# from funasr.export.models.decoders import XformerDecoderSANM

from funasr.models.e2e_asr_paraformer import Paraformer
from funasr.export.models.e2e_asr_paraformer import Paraformer as Paraformer_export

def get_model(model, export_config=None):

    if isinstance(model, Paraformer):
        return Paraformer_export(model, **export_config)
    else:
        raise "The model is not exist!"


# def get_encoder(model, frontend, preencoder, predictor=None, export_config=None):
#     if isinstance(model, espnetRNNEncoder) or isinstance(model, espnetVGGRNNEncoder):
#         return RNNEncoder(model, frontend, preencoder, **export_config)
#     elif isinstance(model, espnetContextualTransformer) or isinstance(model, espnetContextualConformer):
#         return ContextualBlockXformerEncoder(model, **export_config)
#     elif isinstance(model, espnetTransformerEncoder):
#         return TransformerEncoder(model, frontend, preencoder, **export_config)
#     elif isinstance(model, espnetConformerEncoder):
#         return ConformerEncoder(model, frontend, preencoder, **export_config)
#     elif isinstance(model, SANMEncoder_tf) or isinstance(model, SANMEncoderChunkOpt_tf):
#         return TransformerEncoderSANM_tf(model, frontend, preencoder, predictor, **export_config)
#     else:
#         raise "The model is not exist!"


#
# def get_decoder(model, export_config):
#     if isinstance(model, espnetRNNDecoder):
#         return RNNDecoder(model, **export_config)
#     elif isinstance(model, espnetTransducerDecoder):
#         return TransducerDecoder(model, **export_config)
#     elif isinstance(model, FsmnDecoderSCAMAOpt_tf):
#         return XformerDecoderSANM(model, **export_config)
#     else:
#         return XformerDecoder(model, **export_config)
#
#
# def get_lm(model, export_config):
#     if isinstance(model, espnetSequentialRNNLM):
#         return SequentialRNNLM(model, **export_config)
#     elif isinstance(model, espnetTransformerLM):
#         return TransformerLM(model, **export_config)
#
#
# def get_frontend_models(model, export_config):
#     if isinstance(model, espnetS3PRLModel):
#         return S3PRLModel(model, **export_config)
#     else:
#         return None
#
    