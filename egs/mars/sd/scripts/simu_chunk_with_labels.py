import logging
import numpy as np
import soundfile
import kaldiio
from funasr.utils.job_runner import MultiProcessRunnerV3
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import os
import argparse
from collections import OrderedDict
import random


class MyRunner(MultiProcessRunnerV3):

    def prepare(self, parser: argparse.ArgumentParser):
        parser.add_argument("--label_scp", type=str, required=True)
        parser.add_argument("--wav_scp", type=str, required=True)
        parser.add_argument("--utt2spk", type=str, required=True)
        parser.add_argument("--spk2meeting", type=str, required=True)
        parser.add_argument("--utt2xvec", type=str, required=True)
        parser.add_argument("--out_dir", type=str, required=True)
        parser.add_argument("--chunk_size", type=int, default=16)
        parser.add_argument("--chunk_shift", type=int, default=4)
        parser.add_argument("--frame_shift", type=float, default=0.01)
        args = parser.parse_args()

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        label_list = load_scp_as_list(args.label_scp)
        wav_scp = load_scp_as_dict(args.wav_scp)
        utt2spk = load_scp_as_dict(args.utt2spk)
        utt2xvec = load_scp_as_dict(args.utt2xvec)
        spk2meeting = load_scp_as_dict(args.spk2meeting)

        meeting2spks = OrderedDict()
        for spk, meeting in spk2meeting.items():
            if meeting not in meeting2spks:
                meeting2spks[meeting] = []
            meeting2spks[meeting].append(spk)

        spk2utts = OrderedDict()
        for utt, spk in utt2spk.items():
            if spk not in spk2utts:
                spk2utts[spk] = []
            spk2utts[spk].append(utt)

        return label_list, (wav_scp, utt2xvec, spk2utts, meeting2spks), args

    def post(self, results_list, args):
        pass


def process(task_args):
    task_idx, task_list, (wav_scp, utt2xvec, spk2utts, meeting2spks), args = task_args
    out_path = os.path.join(args.out_dir, "wav_mix.{}".format(task_idx+1))
    wav_mix_writer = kaldiio.WriteHelper('ark,scp:{}.ark,{}.scp'.format(out_path, out_path))

    out_path = os.path.join(args.out_dir, "wav_sep.{}".format(task_idx + 1))
    wav_sep_writer = kaldiio.WriteHelper('ark,scp:{}.ark,{}.scp'.format(out_path, out_path))

    out_path = os.path.join(args.out_dir, "label.{}".format(task_idx + 1))
    label_writer = kaldiio.WriteHelper('ark,scp:{}.ark,{}.scp'.format(out_path, out_path))

    idx = 0
    for _, label_path in task_list:
        rand_shift = random.randint(0, int(args.chunk_shift / args.frame_shift))
        whole_label = kaldiio.load_mat(label_path)
        whole_label = whole_label[rand_shift:]
        num_chunk = (whole_label.shape[0] - args.chunk_size) // args.chunk_shift
        for i in range(num_chunk):
            utt_id = "part{}_utt{:10d}".format(task_idx + 1, idx + 1)


    wav_mix_writer.close()
    wav_sep_writer.close()
    label_writer.close()
    return None


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
