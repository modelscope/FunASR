import os
import shutil

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.hub.snapshot_download import snapshot_download

def modelscope_infer_after_finetune(params):
    # prepare for decoding

    try:
        pretrained_model_path = snapshot_download(params["modelscope_model_name"], cache_dir=params["output_dir"])
    except BaseException:
        raise BaseException(f"Please download pretrain model from ModelScope firstly.")
    shutil.copy(os.path.join(params["output_dir"], params["decoding_model_name"]), os.path.join(pretrained_model_path, "model.pb"))
    decoding_path = os.path.join(params["output_dir"], "decode_results")
    if os.path.exists(decoding_path):
        shutil.rmtree(decoding_path)
    os.mkdir(decoding_path)

    # decoding
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model=pretrained_model_path,
        output_dir=decoding_path,
        batch_size=params["batch_size"]
    )
    audio_in = os.path.join(params["data_dir"], "wav.scp")
    inference_pipeline(audio_in=audio_in)

    # computer CER if GT text is set
    text_in = os.path.join(params["data_dir"], "text")
    if os.path.exists(text_in):
        text_proc_file = os.path.join(decoding_path, "1best_recog/text")
        compute_wer(text_in, text_proc_file, os.path.join(decoding_path, "text.cer"))


if __name__ == '__main__':
    import sys
    
    model = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    checkpoint_name = sys.argv[3]
    
    try:
        pretrained_model_path = snapshot_download(model, cache_dir=checkpoint_dir)
    except BaseException:
        raise BaseException(f"Please download pretrain model from ModelScope firstly.")
    shutil.copy(os.path.join(checkpoint_dir, checkpoint_name), os.path.join(pretrained_model_path, "model.pb"))
