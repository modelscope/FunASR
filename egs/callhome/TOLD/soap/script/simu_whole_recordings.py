import argparse
import numpy as np
import librosa
import soundfile as sf
import os
import random
import json
from funasr.utils.misc import load_scp_as_dict, load_scp_as_list
from tqdm import tqdm


def mix_wav_noise(wav, noise, snr):
    n_repeat = len(wav) // len(noise) + 1
    noise = np.repeat(noise, n_repeat, axis=0)
    st = random.randint(0, len(noise) - len(wav))
    noise = noise[st: st+len(wav)]

    wav_mag = np.linalg.norm(wav, ord=2)
    noise_mag = np.linalg.norm(noise, ord=2)
    scale = wav_mag / (10 ** (float(snr) / 20))
    noise = noise / noise_mag * scale
    check_snr = 20 * np.log10(np.linalg.norm(wav, ord=2) / np.linalg.norm(noise, ord=2))
    if abs(check_snr - snr) >= 1e-2:
        print("SNR: {:.4f}, real SNR: {:.4f}".format(snr, check_snr))
    return wav + noise


def calc_labels(rttms, args):
    turns = []
    total_length = 0
    for spk, st, dur in rttms:
        if args.random_interval:
            # random shift the interval with 20% of duration
            x = random.uniform(-dur*0.2, dur*0.2)
            st = max(0, st + x)
            # random squeeze or extend the interval
            dur += random.uniform(-dur*0.5, dur*0.5)
        if st + dur > total_length:
            total_length = st + dur
        turns.append([spk, st, dur])

    # resort the turns according start point
    turns = sorted(turns, key=lambda x: x[1])

    spk_list = []
    for spk, st, dur in turns:
        if spk not in spk_list:
            spk_list.append(spk)

    total_length = int(total_length * args.sr)
    labels = np.zeros((len(spk_list), total_length), float)
    for spk, org_st, org_dur in turns:
        # random re-assign speaker to make more various samples
        st, dur = int(org_st * args.sr), int(org_dur * args.sr)
        if args.random_assign_spk:
            spk = random.choice(spk_list)
        spk_id = spk_list.index(spk)
        labels[spk_id, st:st+dur] = 1.0

    new_turns = []
    for i in range(len(spk_list)):
        st = 0
        in_interval = False
        for j in range(total_length):
            if labels[i, j] == 1 and not in_interval:
                in_interval = True
                st = j
            if (labels[i, j] == 0 or j == total_length-1) and in_interval:
                in_interval = False
                new_turns.append((spk_list[i], float(st) / args.sr, float(j - st) / args.sr))
    new_turns = sorted(new_turns, key=lambda x: x[1])

    return labels, spk_list, new_turns


def save_wav(data, wav_path, sr):
    if np.max(np.abs(data)).item() > 32767:
        data = data / np.max(np.abs(data)) * 0.9 * 32767
    sf.write(wav_path, data.astype(np.int16), sr, "PCM_16", "LITTLE", "WAV", True)


def build(mid, meeting2rttm, spk2wav, noise_scp, room2rirs, args):
    mid = "m{:05d}".format(mid+1)
    if args.corpus_name is not None:
        mid = args.corpus_name + "_" + mid
    org_reco_id = random.choice(meeting2rttm.keys())
    rttms = meeting2rttm[org_reco_id]
    labels, org_spk_list, org_turns = calc_labels(rttms, args)
    n_spk = len(org_spk_list)

    expected_length = labels.shape[1]
    meeting_spk_list = random.sample(spk2wav.keys(), n_spk)
    spk_mask = (np.sum(labels, axis=1) > 0).astype(int)
    pos_spk_list = [spk for spk, mask in zip(meeting_spk_list, spk_mask) if mask == 1]
    noise_id, noise_path = random.choice(noise_scp)
    noise_wav = librosa.load(noise_path, args.sr, True)[0] * 32767
    snr = random.choice(args.snr_list)
    room_id = random.choice(room2rirs.keys())
    # different speakers can locate at the same position a.k.a. the same rir.
    rir_list = [random.choice(room2rirs[room_id]) for _ in range(n_spk)]

    mata = {
        "id": mid,
        "num_spk": n_spk,
        "pos_spk": pos_spk_list,
        "spk_list": meeting_spk_list,
        "seg_info": [],
        "noise": noise_id,
        "snr": snr,
        "length": expected_length,
        "meeting_info": org_reco_id,
        "room_id": room_id
    }
    sig = np.zeros((expected_length, ), dtype=np.float32)
    for i, spk in enumerate(meeting_spk_list):
        if spk in pos_spk_list:
            wav = librosa.load(spk2wav[spk], args.sr, True)[0] * 32767
            if len(wav) <= expected_length:
                # NOTE: to repeat an array, use np.tile rather than np.repeats
                wav = np.tile(wav, expected_length // len(wav) + 1)
            spk_st = np.random.randint(0, len(wav) - expected_length)
            spk_sig = wav[spk_st: spk_st+expected_length]
            spk_sig = spk_sig * labels[i, :]
            rir_wav = librosa.load(rir_list[i][1], args.sr, True)[0] * 32767
            spk_sig = np.convolve(spk_sig, rir_wav, "full")[:expected_length]
            mata["seg_info"].append([spk, spk_st, rir_list[i][0]])
            sig += spk_sig

    mix = mix_wav_noise(sig, noise_wav, snr)
    if np.max(np.abs(mix)).item() > 32767:
        mix = mix / np.max(np.abs(mix)) * 0.9 * 32767
    save_path = os.path.join(args.out_dir, "{}.wav".format(mid))
    sf.write(save_path, mix.astype(np.int16), args.sr, "PCM_16", "LITTLE", "WAV", True)

    rttm_file = open(os.path.join(args.out_dir, "{}.rttm".format(mid)), "wt")
    for spk, st, dur in org_turns:
        rttm_file.write("SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>{}".format(
            mid, st, dur, meeting_spk_list[org_spk_list.index(spk)], os.linesep))
    rttm_file.close()

    return mata, mix, labels


def filter_spk_num(meeting2rttm, reco2num_spk, spk_num):
    meeting_list = meeting2rttm.keys()
    filtered_list = list(filter(lambda x: int(reco2num_spk[x]) == spk_num, meeting_list))
    new_dict = {key: meeting2rttm[key] for key in filtered_list}
    print("Keep {} out of {} according to speaker number {}".format(len(new_dict), len(meeting2rttm), spk_num))
    return new_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--total_mix", type=int, default=1)
    parser.add_argument("--sr", type=int, default=8000)
    parser.add_argument("--snr_list", type=int, default=[15, 20, 25], nargs="+")
    parser.add_argument("--spk_num", type=int, default=0)

    parser.add_argument("--corpus_name", type=str, default=None)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--no_bar", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--random_assign_spk", action="store_true", default=False)
    parser.add_argument("--random_interval", action="store_true", default=False)
    args = parser.parse_args()
    assert args.sr == 8000, "For callhome dataset, the sample rate should be 8000, use --sr 8000."

    # SPEAKER iaaa 0 0 1.08 <NA> <NA> B <NA> <NA>
    meeting2rttm = {}
    for one_line in open(os.path.join(args.dir, "ref.rttm")):
        parts = one_line.strip().split(" ")
        mid, spk, st, dur = parts[1], parts[7], float(parts[3]), float(parts[4])
        if mid not in meeting2rttm:
            meeting2rttm[mid] = []
        meeting2rttm[mid].append((spk, st, dur))
    reco2num_spk = load_scp_as_dict(os.path.join(args.dir, "reco2num_spk"))
    if args.spk_num > 1:
        meeting2rttm = filter_spk_num(meeting2rttm, reco2num_spk, args.spk_num)

    spk2wav = load_scp_as_dict(os.path.join(args.dir, "spk2wav.scp"))
    noise_scp = load_scp_as_list(os.path.join(args.dir, "noise.scp"))
    rirs_scp = load_scp_as_list(os.path.join(args.dir, "rirs.scp"))
    room2rirs = {}
    for rir_id, rir_path in rirs_scp:
        room_id = rir_id.rsplit("-", 1)[0]
        if room_id not in room2rirs:
            room2rirs[room_id] = []
        room2rirs[room_id].append((rir_id, rir_path))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    task_list = list(range(args.task_id * args.total_mix, (args.task_id + 1) * args.total_mix))

    mata_data = []
    total = 0
    if args.debug:
        one, wav, label = build(0, meeting2rttm, spk2wav, noise_scp, room2rirs, args)
        mata_data.append(one)
    else:
        for mid in tqdm(task_list, total=len(task_list), ascii=True, disable=args.no_bar):
            one, wav, label = build(mid, meeting2rttm, spk2wav, noise_scp, room2rirs, args)
            mata_data.append(one)
            total += one["length"]
            if args.verbose:
                print("File name: {:20s}, segment num: {:5d}, speaker num: {:2d}, duration: {:7.2f}m".format(
                    one["id"], len(one["seg_info"]), one["num_spk"], float(one["length"]) / args.sr / 60))
    print("Total files: {}, Total duration: {:.2f} hours".format(args.total_mix, (1.0 * total / args.sr / 3600)))
    json.dump(mata_data, open(os.path.join(args.out_dir, "mata.{}.json".format(args.task_id)), "wt"),
              ensure_ascii=False, encoding='utf-8', indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
