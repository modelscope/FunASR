from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import os


def test_wav_cpu_infer():
    output_dir = "./outputs"
    data_path_and_name_and_type = [
        "data/unit_test/test_wav.scp,speech,sound",
        "data/unit_test/test_profile.scp,profile,kaldi_ark",
    ]
    diar_pipeline = pipeline(
        task=Tasks.speaker_diarization,
        model='damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch',
        mode="sond",
        output_dir=output_dir,
        num_workers=0,
        log_level="WARNING",
    )
    results = diar_pipeline(data_path_and_name_and_type)
    print(results)


def test_wav_gpu_infer():
    output_dir = "./outputs"
    data_path_and_name_and_type = [
        "data/unit_test/test_wav.scp,speech,sound",
        "data/unit_test/test_profile.scp,profile,kaldi_ark",
    ]
    diar_pipeline = pipeline(
        task=Tasks.speaker_diarization,
        model='damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch',
        mode="sond",
        output_dir=output_dir,
        num_workers=0,
        log_level="WARNING",
    )
    results = diar_pipeline(data_path_and_name_and_type)
    print(results)


def test_without_profile_gpu_infer():
    raw_inputs = [
        "data/unit_test/raw_inputs/record.wav",
        "data/unit_test/raw_inputs/spk1.wav",
        "data/unit_test/raw_inputs/spk2.wav",
        "data/unit_test/raw_inputs/spk3.wav",
        "data/unit_test/raw_inputs/spk4.wav"
    ]
    diar_pipeline = pipeline(
        task=Tasks.speaker_diarization,
        model='damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch',
        mode="sond_demo",
        num_workers=0,
        log_level="WARNING",
        param_dict={},
    )
    results = diar_pipeline(raw_inputs=raw_inputs)
    print(results)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_wav_cpu_infer()
    test_wav_gpu_infer()
    test_without_profile_gpu_infer()
