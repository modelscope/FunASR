import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="auto-speech-recognition",
        help="task name",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="generic-asr",
    )
    parser.add_argument(
        "--am_model_name",
        type=str,
        default="model.pb",
        help="model file name",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="paraformer",
        help="mode for decoding",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="zh-cn",
        help="language",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--am_model_config",
        type=str,
        default="config.yaml",
        help="config file",
    )
    parser.add_argument(
        "--mvn_file",
        type=str,
        default="am.mvn",
        help="cmvn file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model name",
    )
    parser.add_argument(
        "--pipeline_type",
        type=str,
        default="asr-inference",
        help="pipeline type",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="vocab_size",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output path",
    )
    parser.add_argument(
        "--nat",
        type=str,
        default="",
        help="vocab_size",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="exp1",
        help="model name tag",
    )
    args = parser.parse_args()

    model = {
        "type": args.type,
        "am_model_name": args.am_model_name,
        "model_config": {
            "type": "pytorch",
            "code_base": "funasr",
            "mode": args.mode,
            "lang": args.lang,
            "batch_size": args.batch_size,
            "am_model_config": args.am_model_config,
            "mvn_file": args.mvn_file,
            "model": "speech_{}_asr{}-{}-16k-{}-vocab{}-pytorch-{}".format(args.model_name, args.nat, args.lang,
                                                                           args.dataset, args.vocab_size, args.tag),
        }
    }
    json_dict = {
        "model": model,
        "framework": "pytorch",
        "task": args.task,
        "pipeline": args.pipeline_type,
    }

    with open(os.path.join(args.output_dir, "configuration.json"), "w") as f:
        json.dump(json_dict, f)
