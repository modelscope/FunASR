import logging

import torch

from funasr.layers.global_mvn import GlobalMVN
from funasr.layers.label_aggregation import LabelAggregate, LabelAggregateMaxPooling
from funasr.layers.utterance_mvn import UtteranceMVN
from funasr.models.e2e_diar_eend_ola import DiarEENDOLAModel
from funasr.models.e2e_diar_sond import DiarSondModel
from funasr.models.encoder.conformer_encoder import ConformerEncoder
from funasr.models.encoder.data2vec_encoder import Data2VecEncoder
from funasr.models.encoder.ecapa_tdnn_encoder import ECAPA_TDNN
from funasr.models.encoder.opennmt_encoders.ci_scorers import DotScorer, CosScorer
from funasr.models.encoder.opennmt_encoders.conv_encoder import ConvEncoder
from funasr.models.encoder.opennmt_encoders.fsmn_encoder import FsmnEncoder
from funasr.models.encoder.opennmt_encoders.self_attention_encoder import SelfAttentionEncoder
from funasr.models.encoder.resnet34_encoder import ResNet34Diar, ResNet34SpL2RegDiar
from funasr.models.encoder.rnn_encoder import RNNEncoder
from funasr.models.encoder.sanm_encoder import SANMEncoder, SANMEncoderChunkOpt
from funasr.models.encoder.transformer_encoder import TransformerEncoder
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.fused import FusedFrontends
from funasr.models.frontend.s3prl import S3prlFrontend
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.models.frontend.wav_frontend import WavFrontendMel23
from funasr.models.frontend.windowing import SlidingWindow
from funasr.models.specaug.specaug import SpecAug
from funasr.models.specaug.specaug import SpecAugLFR
from funasr.models.specaug.abs_profileaug import AbsProfileAug
from funasr.models.specaug.profileaug import ProfileAug
from funasr.modules.eend_ola.encoder import EENDOLATransformerEncoder
from funasr.modules.eend_ola.encoder_decoder_attractor import EncoderDecoderAttractor
from funasr.torch_utils.initialize import initialize
from funasr.train.class_choices import ClassChoices

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        wav_frontend=WavFrontend,
        wav_frontend_mel23=WavFrontendMel23,
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
profileaug_choices = ClassChoices(
    name="profileaug",
    classes=dict(
        profileaug=ProfileAug,
    ),
    type_check=AbsProfileAug,
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
label_aggregator_choices = ClassChoices(
    "label_aggregator",
    classes=dict(
        label_aggregator=LabelAggregate,
        label_aggregator_max_pool=LabelAggregateMaxPooling,
    ),
    default=None,
    optional=True,
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        sond=DiarSondModel,
        eend_ola=DiarEENDOLAModel,
    ),
    default="sond",
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        san=SelfAttentionEncoder,
        fsmn=FsmnEncoder,
        conv=ConvEncoder,
        resnet34=ResNet34Diar,
        resnet34_sp_l2reg=ResNet34SpL2RegDiar,
        sanm_chunk_opt=SANMEncoderChunkOpt,
        data2vec_encoder=Data2VecEncoder,
        ecapa_tdnn=ECAPA_TDNN,
        eend_ola_transformer=EENDOLATransformerEncoder,
    ),
    default="resnet34",
)
speaker_encoder_choices = ClassChoices(
    "speaker_encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        rnn=RNNEncoder,
        sanm=SANMEncoder,
        san=SelfAttentionEncoder,
        fsmn=FsmnEncoder,
        conv=ConvEncoder,
        sanm_chunk_opt=SANMEncoderChunkOpt,
        data2vec_encoder=Data2VecEncoder,
    ),
    default=None,
    optional=True
)
cd_scorer_choices = ClassChoices(
    "cd_scorer",
    classes=dict(
        san=SelfAttentionEncoder,
    ),
    default=None,
    optional=True,
)
ci_scorer_choices = ClassChoices(
    "ci_scorer",
    classes=dict(
        dot=DotScorer,
        cosine=CosScorer,
        conv=ConvEncoder,
    ),
    type_check=torch.nn.Module,
    default=None,
    optional=True,
)
# decoder is used for output (e.g. post_net in SOND)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        rnn=RNNEncoder,
        fsmn=FsmnEncoder,
    ),
    type_check=torch.nn.Module,
    default="fsmn",
)
# encoder_decoder_attractor is used for EEND-OLA
encoder_decoder_attractor_choices = ClassChoices(
    "encoder_decoder_attractor",
    classes=dict(
        eda=EncoderDecoderAttractor,
    ),
    type_check=torch.nn.Module,
    default="eda",
)
class_choices_list = [
    # --frontend and --frontend_conf
    frontend_choices,
    # --specaug and --specaug_conf
    specaug_choices,
    # --profileaug and --profileaug_conf
    profileaug_choices,
    # --normalize and --normalize_conf
    normalize_choices,
    # --label_aggregator and --label_aggregator_conf
    label_aggregator_choices,
    # --model and --model_conf
    model_choices,
    # --encoder and --encoder_conf
    encoder_choices,
    # --speaker_encoder and --speaker_encoder_conf
    speaker_encoder_choices,
    # --cd_scorer and cd_scorer_conf
    cd_scorer_choices,
    # --ci_scorer and ci_scorer_conf
    ci_scorer_choices,
    # --decoder and --decoder_conf
    decoder_choices,
    # --eda and --eda_conf
    encoder_decoder_attractor_choices,
]


def build_diar_model(args):
    # token_list
    if args.token_list is not None:
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")
    else:
        token_list = None
        vocab_size = None

    # frontend
    if args.input_size is None:
        frontend_class = frontend_choices.get_class(args.frontend)
        if args.frontend == 'wav_frontend':
            frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
        else:
            frontend = frontend_class(**args.frontend_conf)
    else:
        args.frontend = None
        args.frontend_conf = {}
        frontend = None

    # encoder
    encoder_class = encoder_choices.get_class(args.encoder)
    encoder = encoder_class(**args.encoder_conf)

    if args.model == "sond":
        # data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # Data augmentation for Profiles
        if hasattr(args, "profileaug") and args.profileaug is not None:
            profileaug_class = profileaug_choices.get_class(args.profileaug)
            profileaug = profileaug_class(**args.profileaug_conf)
        else:
            profileaug = None

        # normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # speaker encoder
        if getattr(args, "speaker_encoder", None) is not None:
            speaker_encoder_class = speaker_encoder_choices.get_class(args.speaker_encoder)
            speaker_encoder = speaker_encoder_class(**args.speaker_encoder_conf)
        else:
            speaker_encoder = None

        # ci scorer
        if getattr(args, "ci_scorer", None) is not None:
            ci_scorer_class = ci_scorer_choices.get_class(args.ci_scorer)
            ci_scorer = ci_scorer_class(**args.ci_scorer_conf)
        else:
            ci_scorer = None

        # cd scorer
        if getattr(args, "cd_scorer", None) is not None:
            cd_scorer_class = cd_scorer_choices.get_class(args.cd_scorer)
            cd_scorer = cd_scorer_class(**args.cd_scorer_conf)
        else:
            cd_scorer = None

        # decoder
        decoder_class = decoder_choices.get_class(args.decoder)
        decoder = decoder_class(**args.decoder_conf)

        # logger aggregator
        if getattr(args, "label_aggregator", None) is not None:
            label_aggregator_class = label_aggregator_choices.get_class(args.label_aggregator)
            label_aggregator = label_aggregator_class(**args.label_aggregator_conf)
        else:
            label_aggregator = None

        model_class = model_choices.get_class(args.model)
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            profileaug=profileaug,
            normalize=normalize,
            label_aggregator=label_aggregator,
            encoder=encoder,
            speaker_encoder=speaker_encoder,
            ci_scorer=ci_scorer,
            cd_scorer=cd_scorer,
            decoder=decoder,
            token_list=token_list,
            **args.model_conf,
        )

    elif args.model == "eend_ola":
        # encoder-decoder attractor
        encoder_decoder_attractor_class = encoder_decoder_attractor_choices.get_class(args.encoder_decoder_attractor)
        encoder_decoder_attractor = encoder_decoder_attractor_class(**args.encoder_decoder_attractor_conf)

        # 9. Build model
        model_class = model_choices.get_class(args.model)
        model = model_class(
            frontend=frontend,
            encoder=encoder,
            encoder_decoder_attractor=encoder_decoder_attractor,
            **args.model_conf,
        )

    else:
        raise NotImplementedError("Not supported model: {}".format(args.model))

    # 10. Initialize
    if args.init is not None:
        initialize(model, args.init)

    return model
