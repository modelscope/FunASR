# libtorch, libtorch_fb20, onnx, onnx_dynamic

# wav="rtf_test_data/test/wav_100.scp"
# wav="rtf_test_data/test/wav_1500.scp"
wav="rtf_test_data/test/wav.scp"

for jn in 64 32 1; do
    ./my_test_rtf.sh libtorch $jn $wav
    ./my_test_rtf.sh libtorch_fb20 $jn $wav
    ./my_test_rtf.sh onnx $jn $wav
    ./my_test_rtf.sh onnx_dynamic $jn $wav
    ./my_test_rtf.sh bladedisc $jn $wav
    ./my_test_rtf.sh bladedisc_fp16 $jn $wav
done
