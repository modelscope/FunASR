import os
from funasr.utils.job_runner import MultiProcessRunnerV3
import numpy as np
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
from collections import OrderedDict
from tqdm import tqdm
from scipy.ndimage import median_filter


class MyRunner(MultiProcessRunnerV3):
    def prepare(self, parser):
        parser.add_argument("label_txt", type=str)
        parser.add_argument("map_scp", type=str)
        parser.add_argument("out_rttm", type=str)
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
        utt2labels = sorted(utt2labels, key=lambda x: x[0])
        meeting2map = load_scp_as_dict(args.map_scp)
        meeting2labels = OrderedDict()
        for utt_id, chunk_label in utt2labels:
            mid = utt_id.split("-")[0]
            if mid not in meeting2labels:
                meeting2labels[mid] = []
            meeting2labels[mid].append(chunk_label)
        task_list = [(mid, labels, meeting2map[mid]) for mid, labels in meeting2labels.items()]

        return task_list, None, args

    def post(self, result_list, args):
        with open(args.out_rttm, "wt") as fd:
            for results in result_list:
                fd.writelines(results)


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


def process(task_args):
    _, task_list, _, args = task_args
    spk_list = ["spk{}".format(i+1) for i in range(args.n_spk)]
    template = "SPEAKER {} 1 {:.2f} {:.2f} <NA> <NA> {} <NA> <NA>\n"
    results = []
    for mid, chunk_label_list, map_file_path in tqdm(task_list, total=len(task_list), ascii=True, disable=args.no_pbar):
        utt2map = load_scp_as_list(map_file_path, 'list')
        multi_labels = calc_multi_labels(chunk_label_list, args.chunk_len, args.shift_len, args.n_spk, args.vote_prob)
        multi_labels = smooth_multi_labels(multi_labels, args.smooth_size)
        org_len = sample2ms(int(utt2map[-1][1][1]), args.sr)
        org_multi_labels = np.zeros((org_len, args.n_spk))
        for seg_id, [org_st, org_ed, st, ed] in utt2map:
            org_st, org_dur = sample2ms(int(org_st), args.sr), sample2ms(int(org_ed) - int(org_st), args.sr)
            st, dur = sample2ms(int(st), args.sr), sample2ms(int(ed) - int(st), args.sr)
            ll = min(org_multi_labels[org_st: org_st+org_dur, :].shape[0], multi_labels[st: st+dur, :].shape[0])
            org_multi_labels[org_st: org_st+ll, :] = multi_labels[st: st+ll, :]
        spk_turns = calc_spk_turns(org_multi_labels, spk_list)
        spk_turns = sorted(spk_turns, key=lambda x: x[1])
        for spk, st, dur in spk_turns:
            # TODO: handle the leak of segments at the change points
            if dur > args.ignore_len:
                results.append(template.format(mid, float(st)/100, float(dur)/100, spk))
    return results


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
