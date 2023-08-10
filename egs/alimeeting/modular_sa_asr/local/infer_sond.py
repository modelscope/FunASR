from funasr.bin.diar_inference_launch import inference_launch
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='7'

def main():
    diar_config_path = sys.argv[1] if len(sys.argv) > 1 else "sond_fbank.yaml"
    diar_model_path = sys.argv[2] if len(sys.argv) > 2 else "sond.pb"
    input_dir = sys.argv[3] if len(sys.argv) > 3 else "./inputs"
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "./outputs"
    data_path_and_name_and_type = [
        (input_dir + "/wav.scp", "speech", "sound"),
        (input_dir + "/profile.scp", "profile", "npy"),
    ]
    pipeline = inference_launch(
        mode="sond",
        diar_train_config=diar_config_path,
        diar_model_file=diar_model_path,
        output_dir=output_dir,
        num_workers=16,
        ngpu=1,
    )
    pipeline(data_path_and_name_and_type)


if __name__ == '__main__':
    main()
