import argparse
import logging
import os
from pathlib import Path
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import yaml

from funasr.datasets.collate_fn import CommonCollateFn
from funasr.datasets.preprocessor import CommonPreprocessor
from funasr.layers.abs_normalize import AbsNormalize
from funasr.layers.global_mvn import GlobalMVN
from funasr.layers.utterance_mvn import UtteranceMVN
from funasr.models.ctc import CTC
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.decoder.rnn_decoder import RNNDecoder
from funasr.models.decoder.sanm_decoder import ParaformerSANMDecoder, FsmnDecoderSCAMAOpt
from funasr.models.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from funasr.models.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from funasr.models.decoder.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from funasr.models.decoder.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from funasr.models.decoder.transformer_decoder import ParaformerDecoderSAN
from funasr.models.decoder.transformer_decoder import TransformerDecoder
from funasr.models.decoder.contextual_decoder import ContextualParaformerDecoder
from funasr.models.decoder.transformer_decoder import SAAsrTransformerDecoder
from funasr.models.e2e_asr import ASRModel
from funasr.models.decoder.rnnt_decoder import RNNTDecoder
from funasr.models.joint_net.joint_network import JointNetwork
from funasr.models.e2e_asr_paraformer import Paraformer, ParaformerOnline, ParaformerBert, BiCifParaformer, ContextualParaformer
from funasr.models.e2e_asr_contextual_paraformer import NeatContextualParaformer
from funasr.models.e2e_tp import TimestampPredictor
from funasr.models.e2e_asr_mfcca import MFCCA
from funasr.models.e2e_sa_asr import SAASRModel
from funasr.models.e2e_uni_asr import UniASR
from funasr.models.e2e_asr_transducer import TransducerModel, UnifiedTransducerModel
from funasr.models.e2e_asr_bat import BATModel
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.encoder.conformer_encoder import ConformerEncoder, ConformerChunkEncoder
from funasr.models.encoder.data2vec_encoder import Data2VecEncoder
from funasr.models.encoder.rnn_encoder import RNNEncoder
from funasr.models.encoder.sanm_encoder import SANMEncoder, SANMEncoderChunkOpt
from funasr.models.encoder.transformer_encoder import TransformerEncoder
from funasr.models.encoder.mfcca_encoder import MFCCAEncoder
from funasr.models.encoder.resnet34_encoder import ResNet34Diar
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.default import MultiChannelFrontend
from funasr.models.frontend.fused import FusedFrontends
from funasr.models.frontend.s3prl import S3prlFrontend
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.models.frontend.windowing import SlidingWindow
from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr.models.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,  # noqa: H301
)
from funasr.models.predictor.cif import CifPredictor, CifPredictorV2, CifPredictorV3, BATPredictor
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.models.preencoder.linear import LinearProjection
from funasr.models.preencoder.sinc import LightweightSincConvs
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.models.specaug.specaug import SpecAug
from funasr.models.specaug.specaug import SpecAugLFR
from funasr.modules.subsampling import Conv1dSubsampling
from funasr.tasks.abs_task import AbsTask
from funasr.tokenizer.phoneme_tokenizer import g2p_choices
from funasr.torch_utils.initialize import initialize
from funasr.models.base_model import FunASRModel
from funasr.train.class_choices import ClassChoices
from funasr.train.trainer import Trainer
from funasr.utils.get_default_kwargs import get_default_kwargs
from funasr.utils.nested_dict_action import NestedDictAction
from funasr.utils.types import float_or_none
from funasr.utils.types import int_or_none
from funasr.utils.types import str2bool
from funasr.utils.types import str_or_none

# from funasr.models.paraformer import Paraformer
frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        wav_frontend=WavFrontend,
        multichannelfrontend=MultiChannelFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
        specaug_lfr=SpecAugLFR,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
# specaug_choices = {"specaug":SpecAug}
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
# model_choices = ClassChoices(
#     "model",
#     classes=dict(
#         asr=ASRModel,
#         uniasr=UniASR,
#         paraformer=Paraformer,
#         paraformer_online=ParaformerOnline,
#         paraformer_bert=ParaformerBert,
#         bicif_paraformer=BiCifParaformer,
#         contextual_paraformer=ContextualParaformer,
#         neatcontextual_paraformer=NeatContextualParaformer,
#         mfcca=MFCCA,
#         timestamp_prediction=TimestampPredictor,
#         rnnt=TransducerModel,
#         rnnt_unified=UnifiedTransducerModel,
#         bat=BATModel,
#         sa_asr=SAASRModel,
#     ),
#     type_check=None,
#     default="asr",
# )
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(
        sinc=LightweightSincConvs,
        linear=LinearProjection,
    ),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        sanm_chunk_opt=SANMEncoderChunkOpt,
        data2vec_encoder=Data2VecEncoder,
        mfcca_enc=MFCCAEncoder,
        chunk_conformer=ConformerChunkEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
encoder_choices2 = ClassChoices(
    "encoder2",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        sanm_chunk_opt=SANMEncoderChunkOpt,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
asr_encoder_choices = ClassChoices(
    "asr_encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        sanm_chunk_opt=SANMEncoderChunkOpt,
        data2vec_encoder=Data2VecEncoder,
        mfcca_enc=MFCCAEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
spk_encoder_choices = ClassChoices(
    "spk_encoder",
    classes=dict(
        resnet34_diar=ResNet34Diar,
    ),
    default="resnet34_diar",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(
        hugging_face_transformers=HuggingFaceTransformersPostEncoder,
    ),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
        fsmn_scama_opt=FsmnDecoderSCAMAOpt,
        paraformer_decoder_sanm=ParaformerSANMDecoder,
        paraformer_decoder_san=ParaformerDecoderSAN,
        contextual_paraformer_decoder=ContextualParaformerDecoder,
        sa_decoder=SAAsrTransformerDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)
decoder_choices2 = ClassChoices(
    "decoder2",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
        fsmn_scama_opt=FsmnDecoderSCAMAOpt,
        paraformer_decoder_sanm=ParaformerSANMDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)

rnnt_decoder_choices = ClassChoices(
    "rnnt_decoder",
    classes=dict(
        rnnt=RNNTDecoder,
    ),
    type_check=RNNTDecoder,
    default="rnnt",
)

joint_network_choices = ClassChoices(
    name="joint_network",
    classes=dict(
        joint_network=JointNetwork,
    ),
    default="joint_network",
    optional=True,
)

predictor_choices = ClassChoices(
    name="predictor",
    classes=dict(
        cif_predictor=CifPredictor,
        ctc_predictor=None,
        cif_predictor_v2=CifPredictorV2,
        cif_predictor_v3=CifPredictorV3,
        bat_predictor=BATPredictor,
    ),
    type_check=None,
    default="cif_predictor",
    optional=True,
)
predictor_choices2 = ClassChoices(
    name="predictor2",
    classes=dict(
        cif_predictor=CifPredictor,
        ctc_predictor=None,
        cif_predictor_v2=CifPredictorV2,
    ),
    type_check=None,
    default="cif_predictor",
    optional=True,
)
stride_conv_choices = ClassChoices(
    name="stride_conv",
    classes=dict(
        stride_conv1d=Conv1dSubsampling
    ),
    type_check=None,
    default="stride_conv1d",
    optional=True,
)