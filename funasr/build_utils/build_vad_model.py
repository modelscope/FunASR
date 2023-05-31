import torch

from funasr.models.e2e_vad import E2EVadModel
from funasr.models.encoder.fsmn_encoder import FSMN
from funasr.models.frontend.default import DefaultFrontend
from funasr.models.frontend.fused import FusedFrontends
from funasr.models.frontend.s3prl import S3prlFrontend
from funasr.models.frontend.wav_frontend import WavFrontend, WavFrontendOnline
from funasr.models.frontend.windowing import SlidingWindow
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
        wav_frontend_online=WavFrontendOnline,
    ),
    default="default",
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        fsmn=FSMN,
    ),
    type_check=torch.nn.Module,
    default="fsmn",
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        e2evad=E2EVadModel,
    ),
    default="e2evad",
)

class_choices_list = [
    # --frontend and --frontend_conf
    frontend_choices,
    # --encoder and --encoder_conf
    encoder_choices,
    # --model and --model_conf
    model_choices,
]


def build_vad_model(args):
    # frontend
    if args.input_size is None:
        frontend_class = frontend_choices.get_class(args.frontend)
        if args.frontend == 'wav_frontend':
            frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
        else:
            frontend = frontend_class(**args.frontend_conf)
        input_size = frontend.output_size()
    else:
        args.frontend = None
        args.frontend_conf = {}
        frontend = None
        input_size = args.input_size

    # encoder
    encoder_class = encoder_choices.get_class(args.encoder)
    encoder = encoder_class(**args.encoder_conf)

    model_class = model_choices.get_class(args.model)
    model = model_class(encoder=encoder, vad_post_args=args.vad_post_conf, frontend=frontend)

    # initialize
    if args.init is not None:
        initialize(model, args.init)

    return model
