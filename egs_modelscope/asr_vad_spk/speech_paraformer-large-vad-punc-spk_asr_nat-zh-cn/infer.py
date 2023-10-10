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
        output_dir=args.output_dir,
        param_dict={"decoding_model": args.decoding_mode, "hotword": args.hotword_txt}
    )
    inference_pipeline(audio_in=args.audio_in, batch_size_token=args.batch_size_token)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn")
    parser.add_argument('--audio_in', type=str, default="./data/test/wav.scp")
    parser.add_argument('--output_dir', type=str, default="./results/")
    parser.add_argument('--decoding_mode', type=str, default="normal")
    parser.add_argument('--hotword_txt', type=str, default=None)
    parser.add_argument('--batch_size_token', type=int, default=5000)
    parser.add_argument('--gpuid', type=str, default="0")
    args = parser.parse_args()
    modelscope_infer(args)
