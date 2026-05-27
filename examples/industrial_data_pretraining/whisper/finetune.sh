#!/bin/bash
# Whisper Fine-tuning with FunASR
#
# This script fine-tunes OpenAI Whisper models on custom data using FunASR's training framework.
# Supports: whisper-tiny, whisper-base, whisper-small, whisper-medium, whisper-large-v3
#
# Data format: JSONL with "audio" and "text" fields
# {"key": "utt1", "source": "/path/to/audio.wav", "target": "transcription text"}

export CUDA_VISIBLE_DEVICES=0,1

model_name="Whisper-large-v3"
train_data="data/train.jsonl"
val_data="data/val.jsonl"
output_dir="exp/whisper_finetune"

python -m funasr.bin.train \
    ++model="${model_name}" \
    ++model_conf.hub="openai" \
    ++train_data_set_list="${train_data}" \
    ++valid_data_set_list="${val_data}" \
    ++dataset_conf.batch_size=4 \
    ++dataset_conf.num_workers=4 \
    ++train_conf.output_dir="${output_dir}" \
    ++train_conf.max_epoch=10 \
    ++train_conf.lr=1e-5 \
    ++train_conf.warmup_steps=500 \
    ++optim="adam" \
    ++optim_conf.lr=1e-5 \
    ++scheduler="warmuplr" \
    ++scheduler_conf.warmup_steps=500
