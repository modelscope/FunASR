#!/usr/bin/env python3
import argparse
import logging
import os

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="decoding configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name",
                        type=str,
                        default="speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                        help="model name in modelscope")
    parser.add_argument("--model_revision",
                        type=str,
                        default="v1.0.4",
                        help="model revision in modelscope")
    parser.add_argument("--local_model_path",
                        type=str,
                        default=None,
                        help="local model path, usually for fine-tuning")
    parser.add_argument("--wav_list",
                        type=str,
                        help="input wav list")
    parser.add_argument("--output_file",
                        type=str,
                        help="saving decoding results")
    parser.add_argument(
        "--njob",
        type=int,
        default=1,
        help="The number of jobs for each gpu",
    )
    parser.add_argument(
        "--gpuid_list",
        type=str,
        default="",
        help="The visible gpus",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    args = parser.parse_args()

    # set logging messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.info("Decoding args: {}".format(args))

    # gpu setting
    if args.ngpu > 0:
        jobid = int(args.output_file.split(".")[-1])
        gpuid = args.gpuid_list.split(",")[(jobid - 1) // args.njob]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

    if args.local_model_path is None:
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/{}".format(args.model_name),
            model_revision=args.model_revision)
    else:
        inference_pipeline = pipeline(
            task=Tasks.auto_speech_recognition,
            model=args.local_model_path)


    with open(args.wav_list, 'r') as f_wav:
        wav_lines = f_wav.readlines()

    with open(args.output_file, "w") as f_out:
        for line in wav_lines:
            wav_id, wav_path = line.strip().split()
            logging.info("decoding, utt_id: ['{}']".format(wav_id))
            rec_result = inference_pipeline(audio_in=wav_path)
            if 'text' in rec_result:
                text = rec_result["text"]
            else:
                text = ''
            f_out.write(wav_id + " " + text + "\n")
            logging.info("best hypo: {} \n".format(text))
