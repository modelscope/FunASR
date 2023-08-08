import os
from funasr.utils.job_runner import MultiProcessRunnerV3
import numpy as np
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
from collections import OrderedDict
from tqdm import tqdm
from scipy.ndimage import median_filter
import kaldiio


def load_mid_vad(vad_path):
    mid2segment_list = {}
    for one_line in open(vad_path, "rt"):
        utt_id, mid, start, end = one_line.strip().split(" ")
        start, end = float(start), float(end)
        if mid not in mid2segment_list:
            mid2segment_list[mid] = []
        mid2segment_list[mid].append((utt_id, start, end))

    return mid2segment_list


class MyRunner(MultiProcessRunnerV3):
    def prepare(self, parser):
        parser.add_argument("label_txt", type=str)
        parser.add_argument("oracle_vad", type=str, default=None)
        parser.add_argument("out_rttm", type=str)
        parser.add_argument("--sys_vad_prob", type=str, default=None)
        parser.add_argument("--sys_vad_threshold", type=float, default=None)
        parser.add_argument("--vad_smooth_size", type=int, default=7)
        parser.add_argument("--n_spk", type=int, default=4)
        parser.add_argument("--chunk_len", type=int, default=1600)
        parser.add_argument("--shift_len", type=int, default=400)
        parser.add_argument("--ignore_len", type=int, default=5)
        parser.add_argument("--smooth_size", type=int, default=7)
        parser.add_argument("--vote_prob", type=float, default=0.5)
        args = parser.parse_args()

        if not os.path.exists(os.path.dirname(args.out_rttm)):
            os.makedirs(os.path.dirname(args.out_rttm))

        utt2labels = load_scp_as_list(args.label_txt, 'list')
        utt2vad_prob = []
        if args.sys_vad_prob is not None and os.path.exists(args.sys_vad_prob):
            if args.verbose:
                print("Use system vad ark file {}".format(args.sys_vad_prob))
            for (key, vad_prob), (utt_id, _) in zip(kaldiio.load_ark(args.sys_vad_prob), utt2labels):
                utt2vad_prob.append((utt_id, vad_prob))
            utt2vad_prob = sorted(utt2vad_prob, key=lambda x: x[0])

        utt2labels = sorted(utt2labels, key=lambda x: x[0])
        mid2segment_list = load_mid_vad(args.oracle_vad)
        meeting2labels = OrderedDict()
        for utt_id, chunk_label in utt2labels:
            mid = utt_id.split("-")[0]
            if mid not in meeting2labels:
                meeting2labels[mid] = []
            meeting2labels[mid].append(chunk_label)

        mid2vad_list = {}
        if len(utt2vad_prob) > 0:
            for utt_id, vad_prob in utt2vad_prob:
                mid = utt_id.split("-")[0]
                if mid not in mid2vad_list:
                    mid2vad_list[mid] = []
                mid2vad_list[mid].append(vad_prob)

        task_list = [(mid, labels, mid2segment_list[mid], None) if len(mid2vad_list) == 0 else
                     (mid, labels, mid2segment_list[mid], mid2vad_list[mid])
                     for mid, labels in meeting2labels.items()]

        return task_list, None, args

    def post(self, result_list, args):
        ref_vad_rttm = open(args.out_rttm + ".ref_vad", "wt")
        sys_vad_rttm = open(args.out_rttm + ".sys_vad", "wt")
        for results in result_list:
            for one_result in results:
                ref_vad_rttm.writelines(one_result[0])
                sys_vad_rttm.writelines(one_result[1])
        ref_vad_rttm.close()
        sys_vad_rttm.close()


def int2vec(x, vec_dim=8, dtype=np.int):
    b = ('{:0' + str(vec_dim) + 'b}').format(x)
    # little-endian order: lower bit first
    return (np.array(list(b)[::-1]) == '1').astype(dtype)


def seq2arr(seq, vec_dim=8):
    return np.row_stack([int2vec(int(x), vec_dim) for x in seq])


def sample2ms(sample, sr=16000):
    return int(float(sample) / sr * 100)


def calc_multi_labels(chunk_label_list, chunk_len, shift_len, n_spk, vote_prob=0.5):
    n_chunk = len(chunk_label_list)
    last_chunk_valid_frame = len(chunk_label_list[-1]) - (chunk_len - shift_len)
    n_frame = (n_chunk - 2) * shift_len + chunk_len + last_chunk_valid_frame
    multi_labels = np.zeros((n_frame, n_spk), dtype=float)
    weight = np.zeros((n_frame, 1), dtype=float)
    for i in range(n_chunk):
        raw_label = chunk_label_list[i]
        for k in range(len(raw_label)):
            if raw_label[k] == '<unk>':
                raw_label[k] = raw_label[k-1] if k > 0 else '0'
        chunk_multi_label = seq2arr(raw_label, n_spk)
        chunk_len = chunk_multi_label.shape[0]
        multi_labels[i*shift_len:i*shift_len+chunk_len, :] += chunk_multi_label
        weight[i*shift_len:i*shift_len+chunk_len, :] += 1
    multi_labels = multi_labels / weight  # normalizing vote
    multi_labels = (multi_labels > vote_prob).astype(int)  # voting results
    return multi_labels


def calc_spk_turns(label_arr, spk_list):
    turn_list = []
    length = label_arr.shape[0]
    n_spk = label_arr.shape[1]
    for k in range(n_spk):
        if spk_list[k] == "None":
            continue
        in_utt = False
        start = 0
        for i in range(length):
            if label_arr[i, k] == 1 and in_utt is False:
                start = i
                in_utt = True
            if label_arr[i, k] == 0 and in_utt is True:
                turn_list.append([spk_list[k], start, i - start])
                in_utt = False
        if in_utt:
            turn_list.append([spk_list[k], start, length - start])
    return turn_list


def smooth_multi_labels(multi_label, win_len):
    multi_label = median_filter(multi_label, (win_len, 1), mode="constant", cval=0.0).astype(int)
    return multi_label


def calc_vad_mask(segments, total_len):
    vad_mask = np.zeros((total_len, 1), dtype=int)
    for _, start, end in segments:
        start, end = int(start * 100), int(end * 100)
        vad_mask[start: end] = 1
    return vad_mask


def calc_system_vad_mask(vad_prob_list, total_len, args):
    if vad_prob_list is None:
        return 1

    threshold = args.sys_vad_threshold
    chunk_len = args.chunk_len
    shift_len = args.shift_len
    frame_vad_mask = np.zeros((total_len, 1), dtype=float)
    weight = np.zeros((total_len, 1), dtype=float)
    for i, vad_prob in enumerate(vad_prob_list):
        frame_vad_mask[i * shift_len: i * shift_len + chunk_len] += np.greater(vad_prob, threshold).astype(float)
        weight[i * shift_len: i * shift_len + chunk_len] += 1.0
    frame_vad_mask = np.greater(frame_vad_mask / weight, args.vote_prob)
    frame_vad_mask = frame_vad_mask.astype(int)
    frame_vad_mask = smooth_multi_labels(frame_vad_mask.astype(int), args.vad_smooth_size)
    return frame_vad_mask


def generate_rttm(mid, multi_labels, spk_list, args):
    template = "SPEAKER {} 0 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n"
    spk_turns = calc_spk_turns(multi_labels, spk_list)
    spk_turns = sorted(spk_turns, key=lambda x: x[1])
    results = []
    for spk, st, dur in spk_turns:
        # TODO: handle the leak of segments at the change points
        if dur > args.ignore_len:
            results.append(template.format(mid, float(st) / 100, float(dur) / 100, spk))
    return results


def process(task_args):
    _, task_list, _, args = task_args
    spk_list = ["spk{}".format(i+1) for i in range(args.n_spk)]
    results = []
    for mid, chunk_label_list, segments_list, sys_vad_list in tqdm(task_list, total=len(task_list),
                                                                   ascii=True, disable=args.no_pbar):
        multi_labels = calc_multi_labels(chunk_label_list, args.chunk_len, args.shift_len, args.n_spk, args.vote_prob)
        multi_labels = smooth_multi_labels(multi_labels, args.smooth_size)
        oracle_vad_mask = calc_vad_mask(segments_list, multi_labels.shape[0])
        oracle_vad_rttm = generate_rttm(mid, multi_labels * oracle_vad_mask, spk_list, args)
        system_vad_mask = calc_system_vad_mask(sys_vad_list, multi_labels.shape[0], args)
        system_vad_rttm = generate_rttm(mid, multi_labels * system_vad_mask, spk_list, args)
        results.append([oracle_vad_rttm, system_vad_rttm])
    return results


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
