import sys
import os
import re
import string
from whisper_normalizer.english import EnglishTextNormalizer

english_normalizer = EnglishTextNormalizer()


def normalize_text(srcfn, dstfn):
    with open(srcfn, "r") as f_read, open(dstfn, "w") as f_write:
        all_lines = f_read.readlines()
        for line in all_lines:
            line = line.strip()
            line_arr = line.split()
            key = line_arr[0]
            conts = " ".join(line_arr[1:])
            normalized_conts = english_normalizer(conts)
            f_write.write("{0}\t{1}\n".format(key, normalized_conts))


if __name__ == "__main__":
    srcfn = sys.argv[1]
    dstfn = sys.argv[2]
    normalize_text(srcfn, dstfn)
