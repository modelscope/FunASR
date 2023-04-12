import json
import os
import shutil

from multiprocessing import Pool
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from funasr.utils.compute_wer import compute_wer


def modelscope_infer_after_finetune_core(model_dir, output_dir, split_dir, njob, idx):
    output_dir_job = os.path.join(output_dir, "output.{}".format(idx))
    gpu_id = (int(idx) - 1) // njob
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list[gpu_id])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model=model_dir,
        output_dir=output_dir_job,
        batch_size=1
    )
    audio_in = os.path.join(split_dir, "wav.{}.scp".format(idx))
    inference_pipeline(audio_in=audio_in)

def modelscope_infer_after_finetune(params):
    # prepare for multi-GPU decoding
    model_dir = params["model_dir"]
    pretrained_model_path = os.path.join(os.environ["HOME"], ".cache/modelscope/hub", params["modelscope_model_name"])
    for file_name in params["required_files"]:
        if file_name == "configuration.json":
            with open(os.path.join(pretrained_model_path, file_name)) as f:
                config_dict = json.load(f)
                config_dict["model"]["am_model_name"] = params["decoding_model_name"]
            with open(os.path.join(model_dir, "configuration.json"), "w") as f:
                json.dump(config_dict, f, indent=4, separators=(',', ': '))
        else:
            shutil.copy(os.path.join(pretrained_model_path, file_name),
                        os.path.join(model_dir, file_name))
    ngpu = params["ngpu"]
    njob = params["njob"]
    output_dir = params["output_dir"]
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    split_dir = os.path.join(output_dir, "split")
    os.mkdir(split_dir)
    nj = ngpu * njob
    wav_scp_file = os.path.join(params["data_dir"], "wav.scp")
    with open(wav_scp_file) as f:
        lines = f.readlines()
        num_lines = len(lines)
        num_job_lines = num_lines // nj
    start = 0
    for i in range(nj):
        end = start + num_job_lines
        file = os.path.join(split_dir, "wav.{}.scp".format(str(i + 1)))
        with open(file, "w") as f:
            if i == nj - 1:
                f.writelines(lines[start:])
            else:
                f.writelines(lines[start:end])
        start = end

    p = Pool(nj)
    for i in range(nj):
        p.apply_async(modelscope_infer_after_finetune_core,
                      args=(model_dir, output_dir, split_dir, njob, str(i + 1)))
    p.close()
    p.join()

    # combine decoding results
    best_recog_path = os.path.join(output_dir, "1best_recog")
    os.mkdir(best_recog_path)
    files = ["text", "token", "score"]
    for file in files:
        with open(os.path.join(best_recog_path, file), "w") as f:
            for i in range(nj):
                job_file = os.path.join(output_dir, "output.{}/1best_recog".format(str(i + 1)), file)
                with open(job_file) as f_job:
                    lines = f_job.readlines()
                f.writelines(lines)

    # If text exists, compute CER
    text_in = os.path.join(params["data_dir"], "text")
    if os.path.exists(text_in):
        text_proc_file = os.path.join(best_recog_path, "token")
        compute_wer(text_in, text_proc_file, os.path.join(best_recog_path, "text.cer"))

if __name__ == '__main__':
    params = {}
    params["modelscope_model_name"] = "damo/speech_UniASR_asr_2pass-zh-cn-8k-common-vocab3445-pytorch-offline"
    params["required_files"] = ["am.mvn", "decoding.yaml", "configuration.json"]
    params["model_dir"] = "./checkpoint"
    params["output_dir"] = "./results"
    params["data_dir"] = "./data/test"
    params["decoding_model_name"] = "20epoch.pb"
    params["ngpu"] = 1
    params["njob"] = 1
    modelscope_infer_after_finetune(params)

