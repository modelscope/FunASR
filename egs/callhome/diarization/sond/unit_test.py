from funasr.bin.diar_inference_launch import inference_launch
import os


def test_fbank_cpu_infer():
    diar_config_path = "sond_fbank.yaml"
    diar_model_path = "sond.pb"
    output_dir = "./outputs"
    data_path_and_name_and_type = [
        ("data/unit_test/test_feats.scp", "speech", "kaldi_ark"),
        ("data/unit_test/test_profile.scp", "profile", "kaldi_ark"),
    ]
    pipeline = inference_launch(
        mode="sond",
        diar_train_config=diar_config_path,
        diar_model_file=diar_model_path,
        output_dir=output_dir,
        num_workers=0,
        log_level="INFO",
    )
    results = pipeline(data_path_and_name_and_type)
    print(results)


def test_fbank_gpu_infer():
    diar_config_path = "sond_fbank.yaml"
    diar_model_path = "sond.pb"
    output_dir = "./outputs"
    data_path_and_name_and_type = [
        ("data/unit_test/test_feats.scp", "speech", "kaldi_ark"),
        ("data/unit_test/test_profile.scp", "profile", "kaldi_ark"),
    ]
    pipeline = inference_launch(
        mode="sond",
        diar_train_config=diar_config_path,
        diar_model_file=diar_model_path,
        output_dir=output_dir,
        ngpu=1,
        num_workers=1,
        log_level="INFO",
    )
    results = pipeline(data_path_and_name_and_type)
    print(results)


def test_wav_gpu_infer():
    diar_config_path = "config.yaml"
    diar_model_path = "sond.pb"
    output_dir = "./outputs"
    data_path_and_name_and_type = [
        ("data/unit_test/test_wav.scp", "speech", "sound"),
        ("data/unit_test/test_profile.scp", "profile", "kaldi_ark"),
    ]
    pipeline = inference_launch(
        mode="sond",
        diar_train_config=diar_config_path,
        diar_model_file=diar_model_path,
        output_dir=output_dir,
        ngpu=1,
        num_workers=1,
        log_level="WARNING",
    )
    results = pipeline(data_path_and_name_and_type)
    print(results)


def test_without_profile_gpu_infer():
    diar_config_path = "config.yaml"
    diar_model_path = "sond.pb"
    output_dir = "./outputs"
    raw_inputs = [[
        "data/unit_test/raw_inputs/record.wav",
        "data/unit_test/raw_inputs/spk1.wav",
        "data/unit_test/raw_inputs/spk2.wav",
        "data/unit_test/raw_inputs/spk3.wav",
        "data/unit_test/raw_inputs/spk4.wav"
    ]]
    pipeline = inference_launch(
        mode="sond_demo",
        diar_train_config=diar_config_path,
        diar_model_file=diar_model_path,
        output_dir=output_dir,
        ngpu=1,
        num_workers=1,
        log_level="WARNING",
        param_dict={},
    )
    results = pipeline(raw_inputs=raw_inputs)
    print(results)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    test_fbank_cpu_infer()
    # test_fbank_gpu_infer()
    # test_wav_gpu_infer()
    # test_without_profile_gpu_infer()
