import os
import shutil
from multiprocessing import Pool

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from funasr.utils.compute_wer import compute_wer


def modelscope_infer_core(output_dir, split_dir, njob, idx):
    output_dir_job = os.path.join(output_dir, "output.{}".format(idx))
    gpu_id = (int(idx) - 1) // njob
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list[gpu_id])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    inference_pipline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-aishell2-vocab8404-pytorch",
        output_dir=output_dir_job,
        batch_size=64
    )
    audio_in = os.path.join(split_dir, "wav.{}.scp".format(idx))
    inference_pipline(audio_in=audio_in)


def modelscope_infer(params):
    # prepare for multi-GPU decoding
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
        p.apply_async(modelscope_infer_core,
                      args=(output_dir, split_dir, njob, str(i + 1)))
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


if __name__ == "__main__":
    params = {}
    params["data_dir"] = "./data/test"
    params["output_dir"] = "./results"
    params["ngpu"] = 1
    params["njob"] = 1
    modelscope_infer(params)
