chmod +x fst/*
[ -f path.sh ] && . ./path.sh

# download train corpus and lexicon
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/requirements/lm.tar.gz
tar -zxvf lm.tar.gz

# train lm, make sure that srilm is installed
bash fst/train_lms.sh

# generate lexicon
python3 fst/generate_lexicon.py lm/corpus.dict lm/lexicon.txt lm/lexicon.out 

# Compile the lexicon and token FSTs
fst/compile_dict_token.sh  lm lm/tmp lm/lang

# Compile the language-model FST and the final decoding graph TLG.fst
fst/make_decode_graph.sh lm lm/lang || exit 1;

# Collect resource files required for decoding
fst/collect_resource_file.sh lm lm/resource
