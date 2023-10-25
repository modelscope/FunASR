chmod +x tools/*
[ -f path.sh ] && . ./path.sh
# train lm, make sure that srilm is installed
bash tools/train_lms.sh

# generate lexicon
python3 tools/generate_lexicon.py lm/corpus.dict lm/lexicon.txt lm/lexicon.out 

# Compile the lexicon and token FSTs
tools/compile_dict_token.sh  lm lm/tmp lm/lang

# Compile the language-model FST and the final decoding graph TLG.fst
tools/make_decode_graph.sh lm lm/lang || exit 1;

# Collect resource files required for decoding
tools/collect_resource_file.sh lm lm/resource
