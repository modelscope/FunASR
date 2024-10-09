import sys
from whisper_normalizer.basic import BasicTextNormalizer

basic_normalizer = BasicTextNormalizer()


def normalize_text(srcfn, dstfn):
    with open(srcfn, "r") as f_read, open(dstfn, "w") as f_write:
        all_lines = f_read.readlines()
        for line in all_lines:
            line = line.strip()
            line_arr = line.split()
            if len(line_arr) < 2:
                continue
            key = line_arr[0]
            conts = " ".join(line_arr[1:])
            normalized_conts = basic_normalizer(conts)
            f_write.write("{0}\t{1}\n".format(key, normalized_conts))


if __name__ == "__main__":
    srcfn = sys.argv[1]
    dstfn = sys.argv[2]
    normalize_text(srcfn, dstfn)
