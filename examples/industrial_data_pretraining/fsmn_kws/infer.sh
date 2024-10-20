# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# method1, inference from model hub

model="iic/speech_charctc_kws_phone-xiaoyun"

# for more input type, please ref to readme.md
input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav"

keywords=(小云小云)
keywords_string=$(IFS=,; echo "${keywords[*]}")
echo "keywords: $keywords_string"

python funasr/bin/inference.py \
+model=${model} \
+input=${input} \
+output_dir="./outputs/debug" \
+device="cpu" \
++keywords="\"$keywords_string"\"
