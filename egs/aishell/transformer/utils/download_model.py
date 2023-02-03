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
                        default="damo/speech_data2vec_pretrain-zh-cn-aishell2-16k-pytorch",
                        help="model name in ModelScope")
    args = parser.parse_args()

    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model=args.model_name)
