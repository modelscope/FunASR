from __future__ import print_function
import numpy as np
import os
import sys
import argparse
from funasr.utils.job_runner import MultiProcessRunnerV3
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import librosa
import soundfile as sf
from copy import deepcopy
import json
from tqdm import tqdm


class MyRunner(MultiProcessRunnerV3):
    def prepare(self, parser):
        assert isinstance(parser, argparse.ArgumentParser)
        parser.add_argument("wav_scp", type=str)
        parser.add_argument("rttm_scp", type=str)
        parser.add_argument("out_dir", type=str)
        parser.add_argument("--min_dur", type=float, default=2.0)
        parser.add_argument("--max_spk_num", type=int, default=4)
        args = parser.parse_args()

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        wav_scp = load_scp_as_list(args.wav_scp)
        rttm_scp = load_scp_as_dict(args.rttm_scp)
        task_list = [(mid, wav_path, rttm_scp[mid]) for (mid, wav_path) in wav_scp]
        return task_list, None, args

    def post(self, result_list, args):
        pass


# SPEAKER R8001_M8004_MS801 1 6.90 11.39 <NA> <NA> 1 <NA> <NA>
def calc_multi_label(rttm_path, length, sr=16000, max_spk_num=4):
    labels = np.zeros([max_spk_num, length], int)
    spk_list = []
    for one_line in open(rttm_path, 'rt'):
        parts = one_line.strip().split(" ")
        mid, st, dur, spk_name = parts[1], float(parts[3]), float(parts[4]), parts[7]
        if spk_name.isdigit():
            spk_name = "{}_S{:03d}".format(mid, int(spk_name))
        if spk_name not in spk_list:
            spk_list.append(spk_name)
        st, dur = int(st*sr), int(dur*sr)
        idx = spk_list.index(spk_name)
        labels[idx, st:st+dur] = 1
    return labels, spk_list


def get_nonoverlap_turns(multi_label, spk_list):
    turns = []
    label = np.sum(multi_label, axis=0) == 1
    spk, in_turn, st = None, False, 0
    for i in range(len(label)):
        if not in_turn and label[i]:
            st, in_turn = i, True
            spk = spk_list[np.argmax(multi_label[:, i], axis=0)]
        if in_turn and not label[i]:
            in_turn = False
            turns.append([st, i, spk])
    return turns


def process(task_args):
    task_id, task_list, _, args = task_args
    for mid, wav_path, rttm_path in task_list:
        wav = librosa.load(wav_path, args.sr)[0] * 32767
        multi_label, spk_list = calc_multi_label(rttm_path, len(wav), args.sr, args.max_spk_num)
        turns = get_nonoverlap_turns(multi_label, spk_list)
        count = 1
        for st, ed, spk in tqdm(turns, total=len(turns), ascii=True):
            if (ed - st) >= args.min_dur * args.sr:
                seg = wav[st: ed]
                save_path = os.path.join(args.out_dir, mid, spk, "{}_U{:04d}.wav".format(spk, count))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                sf.write(save_path, seg.astype(np.int16), args.sr, "PCM_16", "LITTLE", "WAV", True)
                count += 1
    return None


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
