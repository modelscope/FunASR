import os
import shutil
import argparse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def modelscope_infer(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model=args.model,
        model_revision=args.model_revision,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    inference_pipeline(audio_in=args.audio_in)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950")
    parser.add_argument('--model_revision', type=str, default="v3.0.0")
    parser.add_argument('--audio_in', type=str, default="./data/test/wav.scp")
    parser.add_argument('--output_dir', type=str, default="./results/")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpuid', type=str, default="0")
    args = parser.parse_args()
    modelscope_infer(args)
