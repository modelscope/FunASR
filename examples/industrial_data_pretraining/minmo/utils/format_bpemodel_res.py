import sentencepiece as spm
import sys
import string


input_file = sys.argv[1]
output_file = sys.argv[2]

vocab_file = "/nfsspeech/beinian.lzr/workspace/datasets/vocab/funasr/chn_jpn_yue_eng_langid/chn_jpn_yue_eng_langid.vocab.funasr"
bpemodel_file = "/nfsspeech/beinian.lzr/workspace/datasets/vocab/funasr/chn_jpn_yue_eng_langid/chn_jpn_yue_eng_langid.bpe.model"

vocab_file = "/nfs/beinian.lzr/workspace/local_dataset/vocab/chn_jpn_yue_eng_aed_ser/chn_jpn_yue_eng_spectok.vocab.funasr"
bpemodel_file = "/nfs/beinian.lzr/workspace/local_dataset/vocab/chn_jpn_yue_eng_aed_ser/chn_jpn_yue_eng_spectok.bpe.model"

vocab_file = "/nfs/beinian.lzr/workspace/local_dataset/vocab/chn_jpn_yue_eng_aed_ser_fix_missing/chn_jpn_yue_eng_spectok_fix.vocab.funasr"
bpemodel_file = "/nfs/beinian.lzr/workspace/local_dataset/vocab/chn_jpn_yue_eng_aed_ser_fix_missing/chn_jpn_yue_eng_spectok_fix.bpe.model"

sp = spm.SentencePieceProcessor()
sp.load(bpemodel_file)

vocab_dct = {}
idx = 0
with open(vocab_file) as f:
    for line in f:
        ch = line.strip()
        vocab_dct[ch] = idx
        idx += 1

output_fout = open(output_file, "w")

with open(input_file) as f:
    for line in f:
        content = line.strip().split(" ", 1)
        if len(content) == 2:
            key = content[0]
            token = content[1].split()
        else:
            key = content[0]
            token = []
        token_int = [vocab_dct[x] for x in token]
        token_int = list(filter(lambda x: x < 20055, token_int))
        text = sp.decode(token_int).lower()
        output_fout.writelines("{} {}\n".format(key, text))
