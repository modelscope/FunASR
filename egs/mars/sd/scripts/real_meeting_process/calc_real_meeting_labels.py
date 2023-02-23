import numpy as np
from funasr.utils.job_runner import MultiProcessRunnerV3
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import os
import librosa
import argparse


class MyRunner(MultiProcessRunnerV3):

    def prepare(self, parser):
        parser.add_argument("dir", type=str)
        parser.add_argument("out_dir", type=str)
        parser.add_argument("--n_spk", type=int, default=4)
        parser.add_argument("--remove_sil", default=False, action="store_true")
        args = parser.parse_args()

        meeting_scp = load_scp_as_dict(os.path.join(args.dir, "meeting.scp"))
        rttm_scp = load_scp_as_list(os.path.join(args.dir, "rttm.scp"))

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        task_list = [(mid, meeting_scp[mid], rttm_path) for mid, rttm_path in rttm_scp]
        return task_list, None, args

    def post(self, results_list, args):
        pass


def calc_labels(spk_turns, spk_list, length, n_spk, remove_sil=False, sr=16000):
    multi_label = np.zeros([n_spk, length], dtype=int)
    for _, st, dur, spk in spk_turns:
        st, dur = int(st * sr), int(dur * sr)
        idx = spk_list.index(spk)
        multi_label[idx, st:st+dur] = 1
    if not remove_sil:
        return multi_label.T

    speech_count = np.sum(multi_label, axis=0)
    idx = np.nonzero(speech_count)[0]
    label = multi_label[:, idx].T
    return label  # (T, N)


def build_labels(wav_path, rttm_path, n_spk, remove_sil=False, sr=16000):
    wav_len = int(librosa.get_duration(filename=wav_path, sr=sr) * sr)
    spk_turns = []
    spk_list = []
    for one_line in open(rttm_path, "rt"):
        parts = one_line.strip().split(" ")
        mid, st, dur, spk = parts[1], float(parts[3]), float(parts[4]), int(parts[7])
        spk = "{}_S{:03d}".format(mid, spk)
        if spk not in spk_list:
            spk_list.append(spk)
        spk_turns.append((mid, st, dur, spk))
    labels = calc_labels(spk_turns, spk_list, wav_len, n_spk, remove_sil)
    return labels


def process(task_args):
    _, task_list, _, args = task_args
    for mid, wav_path, rttm_path in task_list:
        meeting_labels = build_labels(wav_path, rttm_path, args.n_spk, args.remove_sil)
        save_path = os.path.join(args.out_dir, "{}.lbl".format(mid))
        np.save(save_path, meeting_labels.astype(bool))
        print(mid)
    return None


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
