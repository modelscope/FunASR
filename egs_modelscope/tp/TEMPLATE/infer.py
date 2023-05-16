import os
import argparse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def modelscope_infer(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    inference_pipeline = pipeline(
        task=Tasks.speech_timestamp,
        model=args.model,
        model_revision='v1.1.0',
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    if args.output_dir is not None:
        inference_pipeline(audio_in=args.audio_in, text_in=args.text_in)
    else:
        print(inference_pipeline(audio_in=args.audio_in, text_in=args.text_in))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="damo/speech_timestamp_prediction-v1-16k-offline")
    parser.add_argument('--audio_in', type=str, default="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_timestamps.wav")
    parser.add_argument('--text_in', type=str, default="一 个 东 太 平 洋 国 家 为 什 么 跑 到 西 太 平 洋 来 了 呢")
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpuid', type=str, default="0")
    args = parser.parse_args()
    modelscope_infer(args)
