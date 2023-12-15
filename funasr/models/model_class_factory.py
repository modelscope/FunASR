
from funasr.models.normalize.global_mvn import GlobalMVN
from funasr.models.normalize.utterance_mvn import UtteranceMVN
from funasr.models.ctc.ctc import CTC

from funasr.models.transducer.rnn_decoder import RNNDecoder
from funasr.models.sanm.sanm_decoder import ParaformerSANMDecoder, FsmnDecoderSCAMAOpt
from funasr.models.transformer.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from funasr.models.transformer.transformer_decoder import DynamicConvolutionTransformerDecoder
from funasr.models.transformer.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from funasr.models.transformer.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from funasr.models.transformer.transformer_decoder import ParaformerDecoderSAN
from funasr.models.transformer.transformer_decoder import TransformerDecoder
from funasr.models.paraformer.contextual_decoder import ContextualParaformerDecoder
from funasr.models.transformer.transformer_decoder import SAAsrTransformerDecoder

from funasr.models.transducer.rnnt_decoder import RNNTDecoder
from funasr.models.transducer.joint_network import JointNetwork


from funasr.models.conformer.conformer_encoder import ConformerEncoder, ConformerChunkEncoder
from funasr.models.data2vec.data2vec_encoder import Data2VecEncoder
from funasr.models.transducer.rnn_encoder import RNNEncoder
from funasr.models.sanm.sanm_encoder import SANMEncoder, SANMEncoderChunkOpt
from funasr.models.transformer.transformer_encoder import TransformerEncoder
from funasr.models.branchformer.branchformer_encoder import BranchformerEncoder
from funasr.models.e_branchformer.e_branchformer_encoder import EBranchformerEncoder
from funasr.models.mfcca.mfcca_encoder import MFCCAEncoder
from funasr.models.sond.encoder.resnet34_encoder import ResNet34Diar
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.default import MultiChannelFrontend
from funasr.models.frontend.fused import FusedFrontends
from funasr.models.frontend.s3prl import S3prlFrontend
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.models.frontend.windowing import SlidingWindow


from funasr.models.paraformer.cif_predictor import CifPredictor, CifPredictorV2, CifPredictorV3, BATPredictor
from funasr.models.specaug.specaug import SpecAug
from funasr.models.specaug.specaug import SpecAugLFR
from funasr.models.transformer.subsampling import Conv1dSubsampling
from funasr.utils.class_choices import ClassChoices
from funasr.models.fsmn_vad.fsmn_encoder import FSMN

from funasr.models.sond.encoder.ecapa_tdnn_encoder import ECAPA_TDNN
from funasr.models.sond.encoder.conv_encoder import ConvEncoder
from funasr.models.sond.encoder.fsmn_encoder import FsmnEncoder
from funasr.models.sond.encoder.resnet34_encoder import ResNet34Diar, ResNet34SpL2RegDiar

from funasr.models.sond.encoder.conv_encoder import ConvEncoder
from funasr.models.sond.encoder.fsmn_encoder import FsmnEncoder
from funasr.models.eend.encoder_decoder_attractor import EncoderDecoderAttractor
from funasr.models.eend.encoder import EENDOLATransformerEncoder

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
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    default=None,
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
        fsmn=FSMN,
        branchformer=BranchformerEncoder,
        e_branchformer=EBranchformerEncoder,
        resnet34=ResNet34Diar,
        resnet34_sp_l2reg=ResNet34SpL2RegDiar,
        ecapa_tdnn=ECAPA_TDNN,
        eend_ola_transformer=EENDOLATransformerEncoder,
        conv=ConvEncoder,
        resnet34_diar=ResNet34Diar,
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
        sa_decoder=SAAsrTransformerDecoder,
        rnnt=RNNTDecoder,
    ),
    default="transformer",
)


joint_network_choices = ClassChoices(
    name="joint_network",
    classes=dict(
        joint_network=JointNetwork,
    ),
    default="joint_network",
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
    default="cif_predictor",
)

stride_conv_choices = ClassChoices(
    name="stride_conv",
    classes=dict(
        stride_conv1d=Conv1dSubsampling
    ),
    default="stride_conv1d",
)