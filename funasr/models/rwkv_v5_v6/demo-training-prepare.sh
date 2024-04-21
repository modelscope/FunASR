#!/bin/bash
#######################################################################################################################
#
# This will generate the initial model, and save it to the output folder
#
#######################################################################################################################
#
# Please firstly create data folder & Download minipile (1498226207 tokens, around 3GB)
# mkdir -p data
# wget --continue -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
# wget --continue -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
#
#######################################################################################################################
#
# MODEL_TYPE="x052" # x052 => rwkv-5.2 (rwkv-5 final)
MODEL_TYPE="x060" # x060 => rwkv-6.0
# MODEL_TYPE="mamba" # pip install mamba_ssm --upgrade
#
N_LAYER="12"
N_EMBD="768"
#
CTX_LEN="512" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE # set output folder
#
#######################################################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
#
python train.py --wandb "" --proj_dir $PROJ_DIR \
 --data_file "data/minipile" --data_type "binidx" --vocab_size 65536 --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 --my_exit_tokens 1498226207 --magic_prime 2926181 \
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 \
 --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1
