import kaldiio
from tqdm import tqdm
import os
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import numpy as np
import argparse
import random
import scipy.io as sio
import logging
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


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
        logging.warning("{} has only {} frames, but expect {} frames at least, use them all.".format(spk, count, 300))
        short_spk_list.append(spk)

    ivc_list = [kaldiio.load_mat(utt2ivc[utt])[np.newaxis, :] for utt in utt_list]
    ivc = np.concatenate(ivc_list, axis=0)
    ivc = np.mean(ivc, axis=0, keepdims=False)
    return ivc


def process(feat_scp, labels_scp, spk2utt, utt2xvec, utt2frames, args):
    out_prefix = "{}_parts{:02d}".format(args.out, args.task_id)
    logger = logging.Logger(out_prefix, logging.INFO)
    file_handler = logging.FileHandler(out_prefix + ".log", mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    ivc_dim = 256
    chunk_size, chunk_shift = args.chunk_size, args.chunk_shift
    label_weights = 2 ** np.array(list(range(args.n_spk)))
    feat_writer = kaldiio.WriteHelper(f"ark,scp,f:{out_prefix}_feat.ark,{out_prefix}_feat.scp")
    ivc_writer = kaldiio.WriteHelper(f"ark,scp,f:{out_prefix}_xvec.ark,{out_prefix}_xvec.scp")
    label_writer = kaldiio.WriteHelper(f"ark,scp,f:{out_prefix}_label.ark,{out_prefix}_label.scp")
    train_spk_list = list(spk2utt.keys())

    frames_list = []
    non_present_spk_list = []
    for mid, feat_path in tqdm(feat_scp, total=len(feat_scp), ascii=True, disable=args.no_pbar):
        if mid not in labels_scp:
            continue
        feat = kaldiio.load_mat(feat_path)
        data = sio.loadmat(labels_scp[mid])
        labels, meeting_spk_list = data["labels"].astype(int), [x.strip() for x in data["spk_list"]]
        if args.add_mid_to_speaker:
            meeting_spk_list = ["{}_{}".format(mid, x) if not x.startswith(mid) else x for x in meeting_spk_list]
        if labels.shape[0] != feat.shape[0]:
            min_len = min(labels.shape[0], feat.shape[0])
            labels, feat = labels[:min_len], feat[:min_len]
            logger.warning("{}: The expected frame_len is {}, but got {}, clip both to {}".format(
                mid, labels.shape[0], feat.shape[0], min_len))
        num_frame = feat.shape[0]
        num_chunk = int(np.ceil(float(num_frame - chunk_size) / chunk_shift)) + 1
        for i in range(num_chunk):
            st, ed = i*chunk_shift, i*chunk_shift+chunk_size
            utt_id = "{}-{:05d}-{:05d}".format(mid, st, ed)
            chunk_feat = feat[st: ed, :]
            chunk_label = labels[st: ed, :]
            frame_pad = chunk_size - chunk_label.shape[0]
            spk_pad = args.n_spk - chunk_label.shape[1]
            chunk_feat = np.pad(chunk_feat, [(0, frame_pad), (0, 0)], "constant", constant_values=0)
            chunk_label = np.pad(chunk_label, [(0, frame_pad), (0, spk_pad)], "constant", constant_values=0)

            feat_writer(utt_id, chunk_feat)

            spk_idx = list(range(max(args.n_spk, len(meeting_spk_list))))
            spk_list = []
            if args.mode == "train":
                random.shuffle(spk_idx)

                if args.n_spk > len(meeting_spk_list):
                    n = random.randint(len(meeting_spk_list), args.n_spk)
                    spk_list.extend(meeting_spk_list)
                    while len(spk_list) < n:
                        spk = random.choice(train_spk_list)
                        if spk not in spk_list:
                            spk_list.append(spk)
                    spk_list.extend(["None"] * (args.n_spk - n))
                else:
                    raise ValueError("Argument n_spk is too small ({} < {}).".format(args.n_spk, len(meeting_spk_list)))
            else:
                spk_list.extend(meeting_spk_list)
                spk_list.extend(["None"] * max(args.n_spk - len(meeting_spk_list), 0))

            xvec_list = []
            for i, spk in enumerate(spk_list):
                if spk == "None":
                    spk_xvec = np.zeros((ivc_dim,), dtype=np.float32)
                elif spk not in spk2utt:
                    # speaker with very short duration
                    spk_xvec = np.zeros((ivc_dim,), dtype=np.float32)
                    # chunk_label[:, i] = 0
                    if spk not in non_present_spk_list:
                        logging.warning("speaker {}/{} does not appear in spk2utt, since it has very short duration.".format(mid, spk))
                        non_present_spk_list.append(spk)
                else:
                    spk_xvec = calc_rand_ivc(spk, spk2utt, utt2xvec, utt2frames, 3000)[np.newaxis, :]
                xvec_list.append(spk_xvec)
            xvec = np.row_stack(xvec_list)
            # shuffle speaker embedding according spk_idx
            xvec = xvec[spk_idx, :]
            ivc_writer(utt_id, xvec)

            # shuffle labels according spk_idx
            feat_label = chunk_label[:, spk_idx]
            # feat_label = np.sum(feat_label * label_weights[np.newaxis, :chunk_label.shape[1]], axis=1).astype(str).tolist()
            label_writer(utt_id, feat_label.astype(np.float32))

        frames_list.append((mid, feat.shape[0]))

        logger.info("{:30s}: {:6d} frames split into {:3d} chunks.".format(mid, num_frame, num_chunk))
    return frames_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=str, default=None,
                        help="feats.scp")
    parser.add_argument("--out", required=True, type=str, default=None,
                        help="The prefix of dumpped files.")
    parser.add_argument("--n_spk", type=int, default=16)
    parser.add_argument("--use_lfr", default=False, action="store_true")
    parser.add_argument("--no_pbar", default=False, action="store_true")
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--chunk_size", type=int, default=1600)
    parser.add_argument("--chunk_shift", type=int, default=400)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--task_size", type=int, default=-1)
    parser.add_argument("--add_mid_to_speaker", type=bool, default=False)
    args = parser.parse_args()
    assert args.sr == 8000, "For callhome dataset, the sample rate should be 8000, use --sr 8000."

    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out))

    feat_scp = load_scp_as_list(os.path.join(args.dir, "feats.scp"))
    if args.task_size > 0:
        feat_scp = feat_scp[args.task_size*args.task_id: args.task_size*(args.task_id+1)]
    labels_scp = load_scp_as_dict(os.path.join(args.dir, "labels.scp"))
    utt2spk = load_scp_as_dict(os.path.join(args.dir, "utt2spk"))
    utt2xvec = load_scp_as_dict(os.path.join(args.dir, "utt2xvec"))
    utt2frames = load_scp_as_dict(os.path.join(args.dir, "utt2num_frames"))

    spk2utt = {}
    for utt, spk in utt2spk.items():
        if utt in utt2xvec and utt in utt2frames and int(utt2frames[utt]) > 25:
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(utt)
    logging.info("Obtain {} speakers.".format(len(spk2utt)))
    logging.info("Task {:02d}: start dump {} meetings.".format(args.task_id, len(feat_scp)))
    # random.shuffle(feat_scp)
    meeting_lens = process(feat_scp, labels_scp, spk2utt, utt2xvec, utt2frames, args)
    total_frames = sum([x[1] for x in meeting_lens])
    logging.info("Total meetings: {:6d}, total frames: {:10d}".format(len(meeting_lens), total_frames))


if __name__ == '__main__':
    main()
