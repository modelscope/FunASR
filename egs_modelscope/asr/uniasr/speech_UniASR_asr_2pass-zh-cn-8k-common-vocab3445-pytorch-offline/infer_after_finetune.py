import json
import os
import shutil

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from funasr.utils.compute_wer import compute_wer


def modelscope_infer_after_finetune(params):
    # prepare for decoding
    pretrained_model_path = os.path.join(os.environ["HOME"], ".cache/modelscope/hub", params["modelscope_model_name"])
    for file_name in params["required_files"]:
        if file_name == "configuration.json":
            with open(os.path.join(pretrained_model_path, file_name)) as f:
                config_dict = json.load(f)
                config_dict["model"]["am_model_name"] = params["decoding_model_name"]
            with open(os.path.join(params["output_dir"], "configuration.json"), "w") as f:
                json.dump(config_dict, f, indent=4, separators=(',', ': '))
        else:
            shutil.copy(os.path.join(pretrained_model_path, file_name),
                        os.path.join(params["output_dir"], file_name))
    decoding_path = os.path.join(params["output_dir"], "decode_results")
    if os.path.exists(decoding_path):
        shutil.rmtree(decoding_path)
    os.mkdir(decoding_path)

    # decoding
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model=params["output_dir"],
        output_dir=decoding_path,
        batch_size=1
    )
    audio_in = os.path.join(params["data_dir"], "wav.scp")
    inference_pipeline(audio_in=audio_in, param_dict={"decoding_model": "normal"})

    # computer CER if GT text is set
    text_in = os.path.join(params["data_dir"], "text")
    if os.path.exists(text_in):
        text_proc_file = os.path.join(decoding_path, "1best_recog/token")
        compute_wer(text_in, text_proc_file, os.path.join(decoding_path, "text.cer"))


if __name__ == '__main__':
    params = {}
    params["modelscope_model_name"] = "damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-offline"
    params["required_files"] = ["am.mvn", "decoding.yaml", "configuration.json"]
    params["output_dir"] = "./checkpoint"
    params["data_dir"] = "./data/test"
    params["decoding_model_name"] = "20epoch.pth"
    modelscope_infer_after_finetune(params)
