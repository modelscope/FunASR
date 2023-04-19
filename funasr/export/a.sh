model_path="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
output_dir="../export"

# [torchscript]
# [torchscript]
python export_model.py --model-name $model_path --export-dir $output_dir --type torch --device cuda
# python export_model.py --model-name $model_path --export-dir $output_dir --type torch
# python export_model.py --model-name $model_path --export-dir $output_dir --type torch --quantize --fallback 20 --audio_in rtf_test_data/test/wav_1500.scp --calib_num 200

# [onnx]
# python export_model.py $model_path $output_dir true true
