import soundfile
import kaldiio
from tqdm import tqdm
import json
import os
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import numpy as np
import argparse
import random

short_spk_list = []
def calc_rand_ivc(spk, spk2utt, utt2ivc, utt2frames, total_len=3000):
    all_utts = spk2utt[spk]
    idx_list = list(range(len(all_utts)))
    random.shuffle(idx_list)
    count = 0
    utt_list = []
    for i in idx_list:
        utt_id = all_utts[i]
        utt_list.append(utt_id)
        count += int(utt2frames[utt_id])
        if count >= total_len:
            break
    if count < 300 and spk not in short_spk_list:
        print("Speaker {} has only {} frames, but expect {} frames at least, use them all.".format(spk, count, 300))
        short_spk_list.append(spk)

    ivc_list = [kaldiio.load_mat(utt2ivc[utt])[np.newaxis, :] for utt in utt_list]
    ivc = np.concatenate(ivc_list, axis=0)
    ivc = np.mean(ivc, axis=0, keepdims=False)
    return ivc


def process(meeting_scp, labels_scp, spk2utt, utt2xvec, utt2frames, meeting2spk_list, args):
    out_prefix = args.out

    ivc_dim = 512
    win_len, win_shift = 400, 160
    label_weights = 2 ** np.array(list(range(args.n_spk)))
    wav_writer = kaldiio.WriteHelper("ark,scp:{}_wav.ark,{}_wav.scp".format(out_prefix, out_prefix))
    ivc_writer = kaldiio.WriteHelper("ark,scp:{}_profile.ark,{}_profile.scp".format(out_prefix, out_prefix))
    label_writer = kaldiio.WriteHelper("ark,scp:{}_label.ark,{}_label.scp".format(out_prefix, out_prefix))


    frames_list = []
    chunk_size = int(args.chunk_size * args.sr)
    chunk_shift = int(args.chunk_shift * args.sr)
    for mid, meeting_wav_path in tqdm(meeting_scp, total=len(meeting_scp), ascii=True, disable=args.no_pbar):
        meeting_wav, sr = soundfile.read(meeting_wav_path, dtype='float32')
        num_chunk = (len(meeting_wav) - chunk_size) // chunk_shift + 1
        meeting_labels = np.load(labels_scp[mid])
        for i in range(num_chunk):
            st, ed = i*chunk_shift, i*chunk_shift+chunk_size
            seg_id = "{}-{:03d}-{:06d}-{:06d}".format(mid, i, int(st/args.sr*100), int(ed/args.sr*100))
            wav_writer(seg_id, meeting_wav[st: ed])

            xvec_list = []
            for spk in meeting2spk_list[mid]:
                spk_xvec = calc_rand_ivc(spk, spk2utt, utt2xvec, utt2frames, 1000)[np.newaxis, :]
                xvec_list.append(spk_xvec)
            for _ in range(args.n_spk - len(xvec_list)):
                xvec_list.append(np.zeros((ivc_dim,), dtype=np.float32))
            xvec = np.row_stack(xvec_list)
            ivc_writer(seg_id, xvec)

            wav_label = meeting_labels[st:ed, :]
            frame_num = (ed-st) // win_shift
            # wav_label = np.pad(wav_label, ((win_len/2, win_len/2), (0, 0)), "constant")
            feat_label = np.zeros((frame_num, wav_label.shape[1]), dtype=int)
            for i in range(frame_num):
                frame_label = wav_label[i*win_shift: (i+1)*win_shift, :]
                feat_label[i, :] = (np.sum(frame_label, axis=0) > 0).astype(int)
            label_writer(seg_id, feat_label)

            frames_list.append((mid, feat_label.shape[0]))
    return frames_list


def calc_spk_list(rttm_path):
    spk_list = []
    for one_line in open(rttm_path, "rt"):
        parts = one_line.strip().split(" ")
        mid, st, dur, spk = parts[1], float(parts[3]), float(parts[4]), int(parts[7])
        spk = "{}_S{:03d}".format(mid, spk)
        if spk not in spk_list:
            spk_list.append(spk)

    return spk_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=str, default=None,
                        help="feats.scp")
    parser.add_argument("--out", required=True, type=str, default=None,
                        help="The prefix of dumpped files.")
    parser.add_argument("--n_spk", type=int, default=4)
    parser.add_argument("--use_lfr", default=False, action="store_true")
    parser.add_argument("--no_pbar", default=False, action="store_true")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--chunk_shift", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out))

    meetings_scp = load_scp_as_list(os.path.join(args.dir, "meetings_rmsil.scp"))
    labels_scp = load_scp_as_dict(os.path.join(args.dir, "labels.scp"))
    rttm_scp = load_scp_as_list(os.path.join(args.dir, "rttm.scp"))
    utt2spk = load_scp_as_dict(os.path.join(args.dir, "utt2spk"))
    utt2xvec = load_scp_as_dict(os.path.join(args.dir, "utt2xvec"))
    utt2wav = load_scp_as_dict(os.path.join(args.dir, "wav.scp"))
    utt2frames = {}
    for uttid, wav_path in utt2wav.items():
        wav, sr = soundfile.read(wav_path, dtype="int16")
        utt2frames[uttid] = int(len(wav) / sr * 100)

    meeting2spk_list = {}
    for mid, rttm_path in rttm_scp:
        meeting2spk_list[mid] = calc_spk_list(rttm_path)

    spk2utt = {}
    for utt, spk in utt2spk.items():
        if utt in utt2xvec and utt in utt2frames and int(utt2frames[utt]) > 25:
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt)

    # random.shuffle(feat_scp)
    meeting_lens = process(meetings_scp, labels_scp, spk2utt, utt2xvec, utt2frames, meeting2spk_list, args)
    total_frames = sum([x[1] for x in meeting_lens])
    print("Total chunks: {:6d}, total frames: {:10d}".format(len(meeting_lens), total_frames))


if __name__ == '__main__':
    main()
