import logging
import numpy as np
import soundfile
import kaldiio
from funasr.utils.job_runner import MultiProcessRunnerV3
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import os
import argparse
from collections import OrderedDict


class MyRunner(MultiProcessRunnerV3):

    def prepare(self, parser: argparse.ArgumentParser):
        parser.add_argument("--rttm_list", type=str, required=True)
        parser.add_argument("--wav_scp_list", type=str, required=True)
        parser.add_argument("--out_dir", type=str, required=True)
        parser.add_argument("--n_spk", type=int, default=8)
        parser.add_argument("--remove_sil", default=False, action="store_true")
        parser.add_argument("--max_overlap", default=0, type=int)
        parser.add_argument("--frame_shift", type=float, default=0.01)
        args = parser.parse_args()

        rttm_list = [x.strip() for x in open(args.rttm_list, "rt", encoding="utf-8").readlines()]
        meeting2rttm = OrderedDict()
        for rttm_path in rttm_list:
            meeting2rttm.update(self.load_rttm(rttm_path))

        wav_scp_list = [x.strip() for x in open(args.wav_scp_list, "rt", encoding="utf-8").readlines()]
        meeting_scp = OrderedDict()
        for scp_path in wav_scp_list:
            meeting_scp.update(load_scp_as_dict(scp_path))

        if len(meeting_scp) != len(meeting2rttm):
            logging.warning("Number of wav and rttm mismatch {} != {}".format(
                len(meeting_scp), len(meeting2rttm)))
            common_keys = set(meeting_scp.keys()) & set(meeting2rttm.keys())
            logging.warning("Keep {} records.".format(len(common_keys)))
            new_meeting_scp = OrderedDict()
            for key in meeting_scp:
                if key not in common_keys:
                    logging.warning("Pop {} from wav scp".format(key))
                else:
                    new_meeting_scp[key] = meeting_scp[key]
            new_meeting2rttm = OrderedDict()
            for key in meeting2rttm:
                if key not in common_keys:
                    logging.warning("Pop {} from rttm scp".format(key))
                else:
                    new_meeting2rttm[key] = meeting2rttm[key]

            meeting_scp, meeting2rttm = new_meeting_scp, new_meeting2rttm
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        task_list = [(mid, meeting_scp[mid], meeting2rttm[mid]) for mid in meeting2rttm.keys()]
        return task_list, None, args

    @staticmethod
    def load_rttm(rttm_path):
        meeting2rttm = OrderedDict()
        for one_line in open(rttm_path, "rt", encoding="utf-8"):
            mid = one_line.strip().split(" ")[1]
            if mid not in meeting2rttm:
                meeting2rttm[mid] = []
            meeting2rttm[mid].append(one_line.strip())

        return meeting2rttm

    def post(self, results_list, args):
        pass


def calc_labels(spk_turns, spk_list, length, n_spk, remove_sil=False, max_overlap=0,
                sr=None, frame_shift=0.01):
    frame_shift = int(frame_shift * sr)
    num_frame = int((float(length) + (float(frame_shift) / 2)) / frame_shift)
    multi_label = np.zeros([n_spk, num_frame], dtype=int)
    for _, st, dur, spk in spk_turns:
        idx = spk_list.index(spk)

        st, dur = int(st * sr), int(dur * sr)
        frame_st = int((float(st) + (float(frame_shift) / 2)) / frame_shift)
        frame_ed = int((float(st+dur) + (float(frame_shift) / 2)) / frame_shift)
        multi_label[idx, frame_st:frame_ed] = 1

    if remove_sil:
        speech_count = np.sum(multi_label, axis=0)
        idx = np.nonzero(speech_count)[0]
        multi_label = multi_label[:, idx]

    if max_overlap > 0:
        speech_count = np.sum(multi_label, axis=0)
        idx = np.nonzero(speech_count <= max_overlap)[0]
        multi_label = multi_label[:, idx]

    label = multi_label.T
    return label  # (T, N)


def build_labels(wav_path, rttms, n_spk, remove_sil=False, max_overlap=0,
                 sr=16000, frame_shift=0.01):
    wav, sr = soundfile.read(wav_path)
    wav_len = len(wav)
    spk_turns = []
    spk_list = []
    for one_line in rttms:
        parts = one_line.strip().split(" ")
        mid, st, dur, spk = parts[1], float(parts[3]), float(parts[4]), parts[7]
        if spk not in spk_list:
            spk_list.append(spk)
        spk_turns.append((mid, st, dur, spk))
    labels = calc_labels(spk_turns, spk_list, wav_len, n_spk, remove_sil, max_overlap, sr, frame_shift)
    return labels, spk_list


def process(task_args):
    task_idx, task_list, _, args = task_args
    spk_list_writer = open(os.path.join(args.out_dir, "spk_list.{}.txt".format(task_idx+1)),
                           "wt", encoding="utf-8")
    out_path = os.path.join(args.out_dir, "labels.{}".format(task_idx + 1))
    label_writer = kaldiio.WriteHelper('ark,scp:{}.ark,{}.scp'.format(out_path, out_path))
    for mid, wav_path, rttms in task_list:
        meeting_labels, spk_list = build_labels(wav_path, rttms, args.n_spk, args.remove_sil, args.max_overlap,
                                                args.sr, args.frame_shift)
        label_writer(mid, meeting_labels)
        spk_list_writer.write("{} {}\n".format(mid, " ".join(spk_list)))

    spk_list_writer.close()
    label_writer.close()
    return None


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
