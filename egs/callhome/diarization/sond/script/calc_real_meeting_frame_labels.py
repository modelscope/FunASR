import numpy as np
from opennmt.utils.job_runner import MultiProcessRunnerV3
from opennmt.utils.misc import load_scp_as_list, load_scp_as_dict
import os
import librosa
import scipy.io as sio
import argparse
from collections import OrderedDict


class MyRunner(MultiProcessRunnerV3):

    def prepare(self, parser):
        parser.add_argument("dir", type=str)
        parser.add_argument("out_dir", type=str)
        parser.add_argument("--n_spk", type=int, default=8)
        parser.add_argument("--remove_sil", default=False, action="store_true")
        parser.add_argument("--frame_shift", type=float, default=0.01)
        args = parser.parse_args()
        assert args.sr == 8000, "For callhome dataset, the sample rate should be 8000, use --sr 8000."

        meeting_scp = load_scp_as_dict(os.path.join(args.dir, "wav.scp"))
        meeting2rttm = self.load_rttm(args.dir)

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        task_list = [(mid, meeting_scp[mid], meeting2rttm[mid]) for mid in meeting2rttm.keys()]
        return task_list, None, args

    def load_rttm(self, dir_path):
        meeting2rttm = OrderedDict()
        if os.path.exists(os.path.join(dir_path, "rttm.scp")):
            rttm_scp = load_scp_as_list(os.path.join(dir_path, "rttm.scp"))
            for mid, rttm_path in rttm_scp:
                meeting2rttm[mid] = []
                for one_line in open(rttm_path, "rt"):
                    meeting2rttm[mid].append(one_line.strip())
        elif os.path.exists(os.path.join(dir_path, "ref.rttm")):
            for one_line in open(os.path.join(dir_path, "ref.rttm"), "rt"):
                mid = one_line.strip().split(" ")[1]
                if mid not in meeting2rttm:
                    meeting2rttm[mid] = []
                meeting2rttm[mid].append(one_line.strip())
        else:
            raise IOError("Neither rttm.scp nor ref.rttm exists in {}".format(dir_path))

        return meeting2rttm

    def post(self, results_list, args):
        pass


def calc_labels(spk_turns, spk_list, length, n_spk, remove_sil=False, sr=8000, frame_shift=0.01):
    frame_shift = int(frame_shift * sr)
    num_frame = int((float(length) + (float(frame_shift) / 2)) / frame_shift)
    multi_label = np.zeros([n_spk, num_frame], dtype=int)
    for _, st, dur, spk in spk_turns:
        idx = spk_list.index(spk)

        st, dur = int(st * sr), int(dur * sr)
        frame_st = int((float(st) + (float(frame_shift) / 2)) / frame_shift)
        frame_ed = int((float(st+dur) + (float(frame_shift) / 2)) / frame_shift)
        multi_label[idx, frame_st:frame_ed] = 1
    if not remove_sil:
        return multi_label.T

    speech_count = np.sum(multi_label, axis=0)
    idx = np.nonzero(speech_count)[0]
    label = multi_label[:, idx].T
    return label  # (T, N)


def build_labels(wav_path, rttms, n_spk, remove_sil=False, sr=8000, frame_shift=0.01):
    wav_len = int(librosa.get_duration(filename=wav_path, sr=sr) * sr)
    spk_turns = []
    spk_list = []
    for one_line in rttms:
        parts = one_line.strip().split(" ")
        mid, st, dur, spk = parts[1], float(parts[3]), float(parts[4]), parts[7]
        if spk not in spk_list:
            spk_list.append(spk)
        spk_turns.append((mid, st, dur, spk))
    labels = calc_labels(spk_turns, spk_list, wav_len, n_spk, remove_sil, sr, frame_shift)
    return labels, spk_list


def process(task_args):
    _, task_list, _, args = task_args
    for mid, wav_path, rttms in task_list:
        meeting_labels, spk_list = build_labels(wav_path, rttms, args.n_spk, args.remove_sil,
                                                args.sr, args.frame_shift)
        save_path = os.path.join(args.out_dir, "{}.lbl".format(mid))
        sio.savemat(save_path, {"labels": meeting_labels.astype(bool), "spk_list": spk_list})
        # print mid
    return None


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
