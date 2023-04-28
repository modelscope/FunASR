if [ $# != 2 ]; then
    echo "./launch_service.sh [model_type] [instance_num]"
    exit
fi
model_type=$1
instance_num=$2

if [ "$model_type" = "onnx" ]; then
    model_repo="model_repo_paraformer_large_offline"
    model_name="model.onnx"
elif [ "$model_type" = "libtorch" ]; then
    model_repo="model_repo_paraformer_torchscritpts"
    model_name="model.torchscripts"
elif [ "$model_type" = "bladedisc_fp16" ]; then
    model_repo="model_repo_paraformer_torchscritpts"
    model_name="model.torchscripts"
else
    echo "Not Supported. [\"onnx\", \"libtorch\", \"bladedisc_fp16\"]"
    exit
fi

rm -f $model_repo/encoder/1/$model_name
rm -f $model_repo/feature_extractor/am.mvn
rm -f $model_repo/feature_extractor/config.yaml
ln -s `realpath export_dir/$model_type/$model_name` $model_repo/encoder/1/$model_name
ln -s `realpath export_dir/$model_type/am.mvn` $model_repo/feature_extractor/am.mvn
ln -s `realpath export_dir/$model_type/config.yaml` $model_repo/feature_extractor/config.yaml

config_file=$model_repo/encoder/config.pbtxt
cp $config_file config.pbtxt
cat config.pbtxt | awk '{gsub(/count: [0-9]/,"count: '$instance_num'"); print}' > $config_file
rm -f config.pbtxt

nvidia-cuda-mps-control -d
tritonserver \
    --model-repository /workspace/$model_repo \
    --pinned-memory-pool-byte-size=512000000 \
    --cuda-memory-pool-byte-size=0:1024000000
