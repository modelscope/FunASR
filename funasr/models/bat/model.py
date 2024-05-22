#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import time
import torch
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Tuple
from distutils.version import LooseVersion

from funasr.register import tables
from funasr.utils import postprocess_utils
from funasr.utils.datadir_writer import DatadirWriter
from funasr.models.transducer.model import Transducer
from funasr.train_utils.device_funcs import force_gatherable
from funasr.models.transformer.scorers.ctc import CTCPrefixScorer
from funasr.losses.label_smoothing_loss import LabelSmoothingLoss
from funasr.models.transformer.scorers.length_bonus import LengthBonus
from funasr.models.transformer.utils.nets_utils import get_transducer_task_io
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr.models.transducer.beam_search_transducer import BeamSearchTransducer


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


@tables.register("model_classes", "BAT")  # TODO: BAT training
class BAT(Transducer):
    pass
