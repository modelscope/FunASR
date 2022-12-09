import os
import sys
import random


in_dir = sys.argv[1]
out_dir = sys.argv[2]
num_split = sys.argv[3]


def split_scp(scp, num):
    assert len(scp) >= num
    avg = len(scp) // num
    out = []
    begin = 0

    for i in range(num):
        if i == num - 1:
            out.append(scp[begin:])
        else:
            out.append(scp[begin:begin+avg])
        begin += avg

    return out


os.path.exists("{}/wav.scp".format(in_dir))
os.path.exists("{}/text".format(in_dir))

with open("{}/wav.scp".format(in_dir), 'r') as infile:
    wav_list = infile.readlines()

with open("{}/text".format(in_dir), 'r') as infile:
    text_list = infile.readlines()

assert len(wav_list) == len(text_list)

x = list(zip(wav_list, text_list))
random.shuffle(x)
wav_shuffle_list, text_shuffle_list = zip(*x)

num_split = int(num_split)
wav_split_list = split_scp(wav_shuffle_list, num_split)
text_split_list = split_scp(text_shuffle_list, num_split)

for idx, wav_list in enumerate(wav_split_list, 1):
    path = out_dir + "/split" + str(num_split) + "/" + str(idx)
    if not os.path.exists(path):
        os.makedirs(path)
    with open("{}/wav.scp".format(path), 'w') as wav_writer:
        for line in wav_list:
            wav_writer.write(line)

for idx, text_list in enumerate(text_split_list, 1):
    path = out_dir + "/split" + str(num_split) + "/" + str(idx)
    if not os.path.exists(path):
        os.makedirs(path)
    with open("{}/text".format(path), 'w') as text_writer:
        for line in text_list:
            text_writer.write(line)
