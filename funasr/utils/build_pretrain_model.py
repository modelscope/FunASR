from funasr.layers.global_mvn import GlobalMVN
from funasr.layers.utterance_mvn import UtteranceMVN
from funasr.models.data2vec import Data2VecPretrainModel
from funasr.models.encoder.data2vec_encoder import Data2VecEncoder
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.windowing import SlidingWindow
from funasr.models.specaug.specaug import SpecAug
from funasr.torch_utils.initialize import initialize
from funasr.train.class_choices import ClassChoices

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(default=DefaultFrontend, sliding_window=SlidingWindow),
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug),
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
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        data2vec_encoder=Data2VecEncoder,
    ),
    default="data2vec_encoder",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        data2vec=Data2VecPretrainModel,
    ),
    default="data2vec",
)
class_choices_list = [
    # --frontend and --frontend_conf
    frontend_choices,
    # --specaug and --specaug_conf
    specaug_choices,
    # --normalize and --normalize_conf
    normalize_choices,
    # --encoder and --encoder_conf
    encoder_choices,
    # --model and --model_conf
    model_choices,
]


def build_pretrain_model(args):
    if args.model_name == "data2vec":
        # frontend
        if args.input_size is None:
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
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
        encoder = encoder_class(
            input_size=input_size,
            **args.encoder_conf,
        )

        model_class = model_choices.get_class("data2vec")
        model = model_class(
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            encoder=encoder,
        )

        # 7. Initialize
        if args.init is not None:
            initialize(model, args.init)

        return model
