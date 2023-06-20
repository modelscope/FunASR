import logging

import torch
from typeguard import check_return_type

from funasr.layers.abs_normalize import AbsNormalize
from funasr.layers.global_mvn import GlobalMVN
from funasr.layers.utterance_mvn import UtteranceMVN
from funasr.models.base_model import FunASRModel
from funasr.models.decoder.abs_decoder import AbsDecoder
from funasr.models.decoder.sv_decoder import DenseDecoder
from funasr.models.e2e_sv import ESPnetSVModel
from funasr.models.encoder.abs_encoder import AbsEncoder
from funasr.models.encoder.resnet34_encoder import ResNet34, ResNet34_SP_L2Reg
from funasr.models.encoder.rnn_encoder import RNNEncoder
from funasr.models.frontend.abs_frontend import AbsFrontend
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.fused import FusedFrontends
from funasr.models.frontend.s3prl import S3prlFrontend
from funasr.models.frontend.wav_frontend import WavFrontend
from funasr.models.frontend.windowing import SlidingWindow
from funasr.models.pooling.statistic_pooling import StatisticPooling
from funasr.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr.models.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,  # noqa: H301
)
from funasr.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr.models.preencoder.linear import LinearProjection
from funasr.models.preencoder.sinc import LightweightSincConvs
from funasr.models.specaug.abs_specaug import AbsSpecAug
from funasr.models.specaug.specaug import SpecAug
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
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
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
model_choices = ClassChoices(
    "model",
    classes=dict(
        espnet=ESPnetSVModel,
    ),
    type_check=FunASRModel,
    default="espnet",
)
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
        resnet34=ResNet34,
        resnet34_sp_l2reg=ResNet34_SP_L2Reg,
        rnn=RNNEncoder,
    ),
    type_check=AbsEncoder,
    default="resnet34",
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
pooling_choices = ClassChoices(
    name="pooling_type",
    classes=dict(
        statistic=StatisticPooling,
    ),
    type_check=torch.nn.Module,
    default="statistic",
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        dense=DenseDecoder,
    ),
    type_check=AbsDecoder,
    default="dense",
)

class_choices_list = [
    # --frontend and --frontend_conf
    frontend_choices,
    # --specaug and --specaug_conf
    specaug_choices,
    # --normalize and --normalize_conf
    normalize_choices,
    # --model and --model_conf
    model_choices,
    # --preencoder and --preencoder_conf
    preencoder_choices,
    # --encoder and --encoder_conf
    encoder_choices,
    # --postencoder and --postencoder_conf
    postencoder_choices,
    # --pooling and --pooling_conf
    pooling_choices,
    # --decoder and --decoder_conf
    decoder_choices,
]


def build_sv_model(args):
    # token_list
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
    logging.info(f"Speaker number: {vocab_size}")

    # 1. frontend
    if args.input_size is None:
        # Extract features in the model
        frontend_class = frontend_choices.get_class(args.frontend)
        frontend = frontend_class(**args.frontend_conf)
        input_size = frontend.output_size()
    else:
        # Give features from data-loader
        args.frontend = None
        args.frontend_conf = {}
        frontend = None
        input_size = args.input_size

    # 2. Data augmentation for spectrogram
    if args.specaug is not None:
        specaug_class = specaug_choices.get_class(args.specaug)
        specaug = specaug_class(**args.specaug_conf)
    else:
        specaug = None

    # 3. Normalization layer
    if args.normalize is not None:
        normalize_class = normalize_choices.get_class(args.normalize)
        normalize = normalize_class(**args.normalize_conf)
    else:
        normalize = None

    # 4. Pre-encoder input block
    # NOTE(kan-bayashi): Use getattr to keep the compatibility
    if getattr(args, "preencoder", None) is not None:
        preencoder_class = preencoder_choices.get_class(args.preencoder)
        preencoder = preencoder_class(**args.preencoder_conf)
        input_size = preencoder.output_size()
    else:
        preencoder = None

    # 5. Encoder
    encoder_class = encoder_choices.get_class(args.encoder)
    encoder = encoder_class(input_size=input_size, **args.encoder_conf)

    # 6. Post-encoder block
    # NOTE(kan-bayashi): Use getattr to keep the compatibility
    encoder_output_size = encoder.output_size()
    if getattr(args, "postencoder", None) is not None:
        postencoder_class = postencoder_choices.get_class(args.postencoder)
        postencoder = postencoder_class(
            input_size=encoder_output_size, **args.postencoder_conf
        )
        encoder_output_size = postencoder.output_size()
    else:
        postencoder = None

    # 7. Pooling layer
    pooling_class = pooling_choices.get_class(args.pooling_type)
    pooling_dim = (2, 3)
    eps = 1e-12
    if hasattr(args, "pooling_type_conf"):
        if "pooling_dim" in args.pooling_type_conf:
            pooling_dim = args.pooling_type_conf["pooling_dim"]
        if "eps" in args.pooling_type_conf:
            eps = args.pooling_type_conf["eps"]
    pooling_layer = pooling_class(
        pooling_dim=pooling_dim,
        eps=eps,
    )
    if args.pooling_type == "statistic":
        encoder_output_size *= 2

    # 8. Decoder
    decoder_class = decoder_choices.get_class(args.decoder)
    decoder = decoder_class(
        vocab_size=vocab_size,
        encoder_output_size=encoder_output_size,
        **args.decoder_conf,
    )

    # 7. Build model
    try:
        model_class = model_choices.get_class(args.model)
    except AttributeError:
        model_class = model_choices.get_class("espnet")
    model = model_class(
        vocab_size=vocab_size,
        token_list=token_list,
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        preencoder=preencoder,
        encoder=encoder,
        postencoder=postencoder,
        pooling_layer=pooling_layer,
        decoder=decoder,
        **args.model_conf,
    )

    # FIXME(kamo): Should be done in model?
    # 8. Initialize
    if args.init is not None:
        initialize(model, args.init)

    assert check_return_type(model)
    return model
