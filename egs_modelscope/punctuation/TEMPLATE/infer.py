import os
import shutil
import argparse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def modelscope_infer(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    inference_pipeline = pipeline(
        task=Tasks.punctuation,
        model=args.model,
        output_dir=args.output_dir,
    )
    inference_pipeline(text_in=args.text_in)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
    parser.add_argument('--text_in', type=str, default="./data/test/punc.txt")
    parser.add_argument('--output_dir', type=str, default="./results/")
    parser.add_argument('--gpuid', type=str, default="0")
    args = parser.parse_args()
    modelscope_infer(args)