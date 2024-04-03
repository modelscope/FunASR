# 如何训练LM

训练脚本详见（[点击此处](../tools/train_compile_ngram.sh)）

## 数据准备
```shell
# 下载: 示例训练语料text、lexicon 和 am建模单元units.txt
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/requirements/lm.tar.gz
# 如果是匹配8k的am模型，使用 https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/requirements/lm_8358.tar.gz
tar -zxvf lm.tar.gz
```

## 训练arpa
```shell
# make sure that srilm is installed
# the format of the text should be:
# BAC009S0002W0122 而 对 楼市 成交 抑制 作用 最 大 的 限 购
# BAC009S0002W0123 也 成为 地方 政府 的 眼中 钉

bash fst/train_lms.sh
```

## 生成lexicon
```shell
python3 fst/generate_lexicon.py lm/corpus.dict lm/lexicon.txt lm/lexicon.out
```

## 编译TLG.fst
编译TLG需要依赖fst的环境，请参考文档安装fts相关环境（[点击此处](../onnxruntime/readme.md)）
```shell

# Compile the lexicon and token FSTs
fst/compile_dict_token.sh  lm lm/tmp lm/lang

# Compile the language-model FST and the final decoding graph TLG.fst
fst/make_decode_graph.sh lm lm/lang || exit 1;

# Collect resource files required for decoding
fst/collect_resource_file.sh lm lm/resource

#编译后的模型资源位于 lm/resource
```

