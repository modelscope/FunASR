import logging

from funasr.layers.global_mvn import GlobalMVN
from funasr.layers.utterance_mvn import UtteranceMVN
from funasr.models.ctc import CTC
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.decoder.contextual_decoder import ContextualParaformerDecoder
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
from funasr.models.e2e_asr import ESPnetASRModel
from funasr.models.e2e_asr_mfcca import MFCCA
from funasr.models.e2e_asr_paraformer import Paraformer, ParaformerBert, BiCifParaformer, ContextualParaformer
from funasr.models.e2e_tp import TimestampPredictor
from funasr.models.e2e_uni_asr import UniASR
from funasr.models.encoder.conformer_encoder import ConformerEncoder
from funasr.models.encoder.data2vec_encoder import Data2VecEncoder
from funasr.models.encoder.mfcca_encoder import MFCCAEncoder
from funasr.models.encoder.rnn_encoder import RNNEncoder
from funasr.models.encoder.sanm_encoder import SANMEncoder, SANMEncoderChunkOpt
from funasr.models.encoder.transformer_encoder import TransformerEncoder
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.default import MultiChannelFrontend
from funasr.models.frontend.fused import FusedFrontends
from funasr.models.frontend.s3prl import S3prlFrontend
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.models.frontend.windowing import SlidingWindow
from funasr.models.predictor.cif import CifPredictor, CifPredictorV2, CifPredictorV3
from funasr.models.specaug.specaug import SpecAug
from funasr.models.specaug.specaug import SpecAugLFR
from funasr.modules.subsampling import Conv1dSubsampling
from funasr.train.class_choices import ClassChoices

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
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
        specaug_lfr=SpecAugLFR,
    ),
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    default=None,
    optional=True,
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        asr=ESPnetASRModel,
        uniasr=UniASR,
        paraformer=Paraformer,
        paraformer_bert=ParaformerBert,
        bicif_paraformer=BiCifParaformer,
        contextual_paraformer=ContextualParaformer,
        mfcca=MFCCA,
        timestamp_prediction=TimestampPredictor,
    ),
    default="asr",
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
    ),
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
    default="rnn",
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
    ),
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
predictor_choices = ClassChoices(
    name="predictor",
    classes=dict(
        cif_predictor=CifPredictor,
        ctc_predictor=None,
        cif_predictor_v2=CifPredictorV2,
        cif_predictor_v3=CifPredictorV3,
    ),
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
    default="cif_predictor",
    optional=True,
)
stride_conv_choices = ClassChoices(
    name="stride_conv",
    classes=dict(
        stride_conv1d=Conv1dSubsampling
    ),
    default="stride_conv1d",
    optional=True,
)


def build_model(args):
    # token_list
    if args.token_list is not None:
        with open(args.token_list) as f:
            token_list = [line.rstrip() for line in f]
        args.token_list = list(token_list)
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")
    else:
        vocab_size = None

    # frontend
    if args.input_size is None:
        # Extract features in the model
        frontend_class = frontend_choices.get_class(args.frontend)
        if args.frontend == 'wav_frontend':
            frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
        else:
            frontend = frontend_class(**args.frontend_conf)
        input_size = frontend.output_size()
    else:
        # Give features from data-loader
        args.frontend = None
        args.frontend_conf = {}
        frontend = None
        input_size = args.input_size

    # data augmentation for spectrogram
    if args.specaug is not None:
        specaug_class = specaug_choices.get_class(args.specaug)
        specaug = specaug_class(**args.specaug_conf)
    else:
        specaug = None

    # normalization layer
    if args.normalize is not None:
        normalize_class = normalize_choices.get_class(args.normalize)
        normalize = normalize_class(**args.normalize_conf)
    else:
        normalize = None

    # encoder
    encoder_class = encoder_choices.get_class(args.encoder)
    encoder = encoder_class(input_size=input_size, **args.encoder_conf)

    # decoder
    decoder_class = decoder_choices.get_class(args.decoder)
    decoder = decoder_class(
        vocab_size=vocab_size,
        encoder_output_size=encoder.output_size(),
        **args.decoder_conf,
    )

    # ctc
    ctc = CTC(
        odim=vocab_size, encoder_output_size=encoder.output_size(), **args.ctc_conf
    )

    
