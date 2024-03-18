# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# To install requirements: pip3 install -U openai-whisper

# method1, inference from model hub

# for more input type, please ref to readme.md
input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"

output_dir="./outputs/debug"

#model="Whisper-small"
#model="Whisper-medium"
model="Whisper-large-v2"
#model="Whisper-large-v3"
hub="openai"

device="cuda:0" # "cuda:0" for gpu0, "cuda:1" for gpu1, "cpu"

python -m funasr.bin.inference \
++model=${model} \
++hub=${hub} \
++input="${input}" \
++output_dir="${output_dir}" \
++device="${device}" \
