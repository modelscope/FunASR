# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)


# which gpu to train or finetune
# export CUDA_VISIBLE_DEVICES="0"
# gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')


export TORCH_DISTRIBUTED_DEBUG=INFO

train_data="/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text_V3/Speech2Text_AlignData_All/PreAlign_Data/20240925_speech2text_v3.0_prealign/20240925_speech2text_v3.0_prealign.json.shuf512.list"
val_data="/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text_V2/Speech2Text_AlignData_All/PreAlign_Data/20240823_speech2text_v2_prealign/json_dir/speech2text_json_shuf.1.head1000.jsonl"


count=$1
gpu_num=$2
suffix=$3

# exp output dir

output_dir="/nfs/beinian.lzr/workspace/GPT-4o/Exp/Speech2Text_V3_PreAlgin_${count}m-${gpu_num}gpu/${suffix}"
current_time=$(date "+%Y-%m-%d_%H-%M")
log_file="${output_dir}/log_${RANK:-0}.${current_time}.txt"


mkdir -p ${output_dir}
echo "log_file: ${log_file}"

workspace=`pwd`
config="MinMo_Speech2Text_PreAlign_8b.yaml"
init_param="/cpfs_speech/zhifu.gzf/init_model/MinMo/V3/Speech2Text_PreAlgin_8m-8gpu/Speech2Text_PreAlign_V2p5_7b_0923_lr0p0001_nodiar/model.pt.ep0.60000"

# gpu_num=4
DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port 26669
"

echo $DISTRIBUTED_ARGS

torchrun $DISTRIBUTED_ARGS \
../../../funasr/bin/train_ds.py \
--config-path "${workspace}/conf" \
--config-name "${config}" \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset="OpenAIDatasetMultiTurn" \
++dataset_conf.index_ds="OpenAIIndexDSJsonl" \
++dataset_conf.data_split_num=512 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.shuffle=true \
++dataset_conf.sort_size=512 \
++dataset_conf.batch_type="token" \
++dataset_conf.batch_size=1500  \
++dataset_conf.batch_size_token_max=8000 \
++dataset_conf.batch_size_sample_max=15 \
++dataset_conf.max_token_length=2048 \
++dataset_conf.max_source_length=8000 \
++dataset_conf.batch_size_scale_threshold=3000 \
++dataset_conf.num_workers=4 \
++dataset_conf.retry=50 \
++train_conf.accum_grad=1 \
++train_conf.max_epoch=10 \
++train_conf.log_interval=100 \
++train_conf.resume=true \
++train_conf.validate_interval=10000 \
++train_conf.save_checkpoint_interval=10000 \
++train_conf.keep_nbest_models=100 \
++train_conf.avg_nbest_model=100 \
++train_conf.use_deepspeed=true \
++train_conf.deepspeed_config="${workspace}/../../deepspeed_conf/ds_stage1.json" \
++init_param=${init_param} \
++output_dir="${output_dir}" 2>&1 | tee ${log_file}


# ++init_param=${init_param} \

