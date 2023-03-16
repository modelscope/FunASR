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
    inference_pipeline(audio_in=audio_in)

    # computer CER if GT text is set
    text_in = os.path.join(params["data_dir"], "text")
    if text_in is not None:
        text_proc_file = os.path.join(decoding_path, "1best_recog/token")
        text_proc_file2 = os.path.join(decoding_path, "1best_recog/token_nosep")
        with open(text_proc_file, 'r') as hyp_reader:
                with open(text_proc_file2, 'w') as hyp_writer:
                    for line in hyp_reader:
                        new_context = line.strip().replace("src","").replace("  "," ").replace("  "," ").strip()
                        hyp_writer.write(new_context+'\n')
        text_in2 = os.path.join(decoding_path, "1best_recog/ref_text_nosep")
        with open(text_in, 'r') as ref_reader:
            with open(text_in2, 'w') as ref_writer:
                for line in ref_reader:
                    new_context = line.strip().replace("src","").replace("  "," ").replace("  "," ").strip()
                    ref_writer.write(new_context+'\n')


        compute_wer(text_in, text_proc_file, os.path.join(decoding_path, "text.sp.cer"))
        compute_wer(text_in2, text_proc_file2, os.path.join(decoding_path, "text.nosp.cer"))

if __name__ == '__main__':
    params = {}
    params["modelscope_model_name"] = "NPU-ASLP/speech_mfcca_asr-zh-cn-16k-alimeeting-vocab4950"
    params["required_files"] = ["feats_stats.npz", "decoding.yaml", "configuration.json"]
    params["output_dir"] = "./checkpoint"
    params["data_dir"] = "./example_data/validation"
    params["decoding_model_name"] = "valid.acc.ave.pb"
    modelscope_infer_after_finetune(params)
