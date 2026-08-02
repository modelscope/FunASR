# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

workspace=`pwd`

# which gpu to train or finetune
export CUDA_VISIBLE_DEVICES="0"
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

# model_name from model_hub, or model_dir in local path
model_name_or_model_dir="FunAudioLLM/Fun-ASR-Nano-2512"

# data dir, which contains: train.json, val.json
train_data=${workspace}/data/train_example.jsonl
val_data=${workspace}/data/val_example.jsonl

# exp output dir
output_dir="./outputs_lora"
log_file="${output_dir}/log.txt"

deepspeed_config=${workspace}/deepspeed_conf/ds_stage1.json

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

# LoRA hyper-parameters. The defaults below match the values shipped in the
# model config (`llm_conf.lora_conf`); override with ++key=value if needed.
#   --llm_conf.lora_conf.r=16
#   --llm_conf.lora_conf.lora_alpha=32
#   --llm_conf.lora_conf.lora_dropout=0.05
#   --llm_conf.lora_conf.target_modules="[q_proj, v_proj]"
#
# Freeze semantics with LoRA:
#   * llm_conf.freeze=true keeps the Qwen3 base weights frozen; only the
#     injected lora_A / lora_B are trainable inside the LLM.
#   * lora_only=true additionally freezes every non-LoRA parameter
#     (audio_encoder / audio_adaptor / ctc_decoder), i.e. pure-LoRA training.
#   * To keep the encoder/adaptor trainable while LoRA-tweaking only the LLM,
#     set lora_only=false and flip the corresponding *_conf.freeze to false.

DISTRIBUTED_ARGS="
    --nnodes ${WORLD_SIZE:-1} \
    --nproc_per_node $gpu_num \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-26669}
"
echo $DISTRIBUTED_ARGS

# funasr trainer path
train_tool=`which funasr-train-ds`
echo "Using funasr trainer: ${train_tool}"

torchrun $DISTRIBUTED_ARGS \
${train_tool} \
++model="${model_name_or_model_dir}" \
++trust_remote_code=true \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=6000  \
++dataset_conf.sort_size=1024 \
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=50 \
++train_conf.log_interval=1 \
++train_conf.resume=true \
++train_conf.validate_interval=2000 \
++train_conf.save_checkpoint_interval=2000 \
++train_conf.effective_save_name_excludes="None" \
++train_conf.keep_nbest_models=20 \
++train_conf.avg_nbest_model=10 \
++train_conf.use_deepspeed=false \
++train_conf.deepspeed_config=${deepspeed_config} \
++optim_conf.lr=0.0001 \
++llm_conf.use_lora=true \
++lora_only=true \
++llm_conf.freeze=true \
++audio_encoder_conf.freeze=true \
++audio_adaptor_conf.freeze=true \
++output_dir="${output_dir}" &> ${log_file}
