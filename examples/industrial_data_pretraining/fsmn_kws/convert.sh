
workspace=`pwd`

# download model
local_path_root=${workspace}/modelscope_models_kws
mkdir -p ${local_path_root}

local_path=${local_path_root}/speech_charctc_kws_phone-xiaoyun
if [ ! -d "$local_path" ]; then
    git clone https://www.modelscope.cn/iic/speech_charctc_kws_phone-xiaoyun.git ${local_path}
fi

export PATH=${local_path}/runtime:$PATH
export LD_LIBRARY_PATH=${local_path}/runtime:$LD_LIBRARY_PATH

config=./conf/fsmn_4e_l10r2_250_128_fdim80_t2599.yaml
torch_nnet=exp/finetune_outputs/model.pt.avg10
out_dir=exp/finetune_outputs

if [ ! -d "$out_dir" ]; then
    mkdir -p $out_dir
fi

python convert.py --config $config --network_file $torch_nnet --model_dir $out_dir --model_name "convert.kaldi.txt" --convert_to kaldi

nnet-copy --binary=true ${out_dir}/convert.kaldi.txt ${out_dir}/convert.kaldi.net
