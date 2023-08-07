import kaldiio
from tqdm import tqdm
import os
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import numpy as np
import argparse
from kaldiio import WriteHelper


def calc_global_ivc(spk, spk2utt, utt2ivc):
    ivc_list = [kaldiio.load_mat(utt2ivc[utt])[np.newaxis, :] for utt in spk2utt[spk]]
    ivc = np.concatenate(ivc_list, axis=0)
    ivc = np.mean(ivc, axis=0, keepdims=False)
    return ivc


def process(idx_scp, spk2utt, utt2xvec, meeting2spk_list, args):
    out_prefix = args.out

    ivc_dim = 256
    print("ivc_dim = {}".format(ivc_dim))
    out_prefix = out_prefix + "_parts00_xvec"
    ivc_writer = WriteHelper(f"ark,scp,f:{out_prefix}.ark,{out_prefix}.scp")
    idx_writer = open(out_prefix + ".idx", "wt")
    spk2xvec = {}
    if args.emb_type == "global":
        print("Use global speaker embedding.")
        for spk in spk2utt.keys():
            spk2xvec[spk] = calc_global_ivc(spk, spk2utt, utt2xvec)[np.newaxis, :]

    frames_list = []
    for utt_id in tqdm(idx_scp, total=len(idx_scp), ascii=True, disable=args.no_pbar):
        mid = utt_id.split("-")[0]
        idx_writer.write(utt_id+"\n")

        xvec_list = []
        for spk in meeting2spk_list[mid]:
            spk_xvec = spk2xvec[spk]
            xvec_list.append(spk_xvec)
        for _ in range(args.n_spk - len(xvec_list)):
            xvec_list.append(np.zeros((ivc_dim,), dtype=np.float32))
        xvec = np.row_stack(xvec_list)
        ivc_writer(utt_id, xvec)

        frames_list.append((mid, 1))
    return frames_list


def calc_spk_list(rttms):
    spk_list = []
    for one_line in rttms:
        parts = [x for x in one_line.strip().split(" ") if x != ""]
        mid, st, dur, spk_name = parts[1], float(parts[3]), float(parts[4]), parts[7]
        spk_name = spk_name.replace("spk", "").replace(mid, "").replace("-", "")
        if spk_name.isdigit():
            spk_name = "{}_S{:03d}".format(mid, int(spk_name))
        else:
            spk_name = "{}_{}".format(mid, spk_name)
        if spk_name not in spk_list:
            spk_list.append(spk_name)

    return spk_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=str, default=None,
                        help="feats.scp")
    parser.add_argument("--out", required=True, type=str, default=None,
                        help="The prefix of dumpped files.")
    parser.add_argument("--n_spk", type=int, default=4)
    parser.add_argument("--no_pbar", default=False, action="store_true")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--emb_type", type=str, default="rand")
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out))

    idx_scp = open(os.path.join(args.dir, "idx"), "r").readlines()
    idx_scp = [x.strip() for x in idx_scp]
    meeting2rttms = {}
    for one_line in open(os.path.join(args.dir, "sys.rttm"), "rt"):
        parts = [x for x in one_line.strip().split(" ") if x != ""]
        mid, st, dur, spk_name = parts[1], float(parts[3]), float(parts[4]), parts[7]
        if mid not in meeting2rttms:
            meeting2rttms[mid] = []
        meeting2rttms[mid].append(one_line)

    utt2spk = load_scp_as_dict(os.path.join(args.dir, "utt2spk"))
    utt2xvec = load_scp_as_dict(os.path.join(args.dir, "utt2xvec"))

    spk2utt = {}
    for utt, spk in utt2spk.items():
        if utt in utt2xvec:
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt)

    meeting2spk_list = {}
    for mid, rttms in meeting2rttms.items():
        meeting2spk_list[mid] = calc_spk_list(rttms)
        new_spk_list = []
        for spk in meeting2spk_list[mid]:
            if spk in spk2utt:
                new_spk_list.append(spk)
        if len(new_spk_list) != len(meeting2spk_list[mid]):
            print("{}: Reduce speaker number from {}(according rttm) to {}(according x-vectors)".format(
                mid, len(meeting2spk_list[mid]), len(new_spk_list)))
        meeting2spk_list[mid] = new_spk_list

    meeting_lens = process(idx_scp, spk2utt, utt2xvec, meeting2spk_list, args)
    print("Total meetings: {:6d}".format(len(meeting_lens)))


if __name__ == '__main__':
    main()
