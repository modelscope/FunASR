import kaldiio
import os
import sys
import numpy as np


def int2vec(x, vec_dim=8, dtype=np.float32):
    b = ('{:0' + str(vec_dim) + 'b}').format(x)
    # little-endian order: lower bit first
    return (np.array(list(b)[::-1]) == '1').astype(dtype)


def seq2arr(seq, vec_dim=8):
    return np.row_stack([int2vec(int(x), vec_dim) for x in seq])


if __name__ == '__main__':
    scp_file = sys.argv[1]
    label_file = sys.argv[2]
    out_file = sys.argv[3]
    max_spk_num = int(sys.argv[4])

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    out_file = out_file.rsplit('.', maxsplit=1)[0]
    wav_writer = kaldiio.WriteHelper("ark,scp,f:{}.ark,{}.scp".format(out_file, out_file))
    for i, (uttid, pse_str) in enumerate(zip(open(scp_file, "rt"), open(label_file, "rt"))):
        uttid, pse_str = uttid.strip().split(" ", maxsplit=1)[0], pse_str.strip()
        bin_label = seq2arr(pse_str.split(" "), vec_dim=max_spk_num)
        wav_writer(uttid, bin_label)

        if i % 100 == 0:
            print(f"Processed {i} samples, the last is {uttid}")

    wav_writer.close()
