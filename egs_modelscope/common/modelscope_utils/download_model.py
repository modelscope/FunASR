#!/usr/bin/env python3
import argparse

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="download model configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name",
                        type=str,
                        default="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                        help="model name in modelscope")
    args = parser.parse_args()

    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model='damo/{}'.format(args.model_name),
        model_revision='v1.0.0')
