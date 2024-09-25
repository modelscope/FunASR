# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method1, inference from model hub

model="iic/speech_sanm_kws_phone-xiaoyun-commands-online"

# for more input type, please ref to readme.md
input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav"

keywords=(小云小云)
keywords_string=$(IFS=,; echo "${keywords[*]}")
echo "keywords: $keywords_string"

python funasr/bin/inference.py \
+model=${model} \
+input=${input} \
+output_dir="./outputs/debug" \
++chunk_size='[4, 8, 4]' \
++encoder_chunk_look_back=0 \
++decoder_chunk_look_back=0 \
+device="cpu" \
++keywords="\"$keywords_string"\"


python funasr/bin/inference.py \
+model=${model} \
+input=${input} \
+output_dir="./outputs/debug" \
++chunk_size='[5, 10, 5]' \
++encoder_chunk_look_back=0 \
++decoder_chunk_look_back=0 \
+device="cpu" \
++keywords="\"$keywords_string"\"
