import os
import shutil
import argparse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def modelscope_infer(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    inference_pipeline = pipeline(
        task=Tasks.voice_activity_detection,
        model=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    inference_pipeline(audio_in=args.audio_in)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch")
    parser.add_argument('--audio_in', type=str, default="./data/test/wav.scp")
    parser.add_argument('--output_dir', type=str, default="./results/")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpuid', type=str, default="0")
    args = parser.parse_args()
    modelscope_infer(args)