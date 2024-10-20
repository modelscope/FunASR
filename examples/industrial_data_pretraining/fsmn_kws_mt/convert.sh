
workspace=`pwd`

# download model
local_path_root=${workspace}/modelscope_models
mkdir -p ${local_path_root}

local_path=${local_path_root}/speech_charctc_kws_phone-xiaoyun_mt
if [ ! -d "$local_path" ]; then
    git clone https://www.modelscope.cn/iic/speech_charctc_kws_phone-xiaoyun_mt.git ${local_path}
fi

export PATH=${local_path}/runtime:$PATH
export LD_LIBRARY_PATH=${local_path}/runtime:$LD_LIBRARY_PATH

# finetune config file
config=./conf/fsmn_4e_l10r2_250_128_fdim80_t2599_t4.yaml

# finetune output checkpoint
torch_nnet=exp/finetune_outputs/model.pt.avg10

out_dir=exp/finetune_outputs

if [ ! -d "$out_dir" ]; then
    mkdir -p $out_dir
fi

python convert.py --config $config \
	--network_file $torch_nnet \
	--model_dir $out_dir \
	--model_name "convert.kaldi.txt" \
	--model_name2 "convert.kaldi2.txt" \
	--convert_to kaldi

nnet-copy --binary=true ${out_dir}/convert.kaldi.txt ${out_dir}/convert.kaldi.net
nnet-copy --binary=true ${out_dir}/convert.kaldi2.txt ${out_dir}/convert.kaldi2.net
