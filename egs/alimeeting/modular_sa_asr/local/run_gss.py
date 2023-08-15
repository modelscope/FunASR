#!/usr/bin/env python
# -- coding: UTF-8

import argparse
import codecs
import glob
import logging
import os
from nara_wpe.utils import stft, istft
import numpy as np
import scipy.io.wavfile as wf
from tqdm import tqdm

from test_gss import *


def get_parser():
    parser = argparse.ArgumentParser("Doing GSS based enhancement.")
    parser.add_argument(
        "--wav-scp",
        type=str,
        required=True,
        help="Wav scp file for enhancement.",
    )
    parser.add_argument(
        "--segments",
        type=str,
        required=True,
        help="Wav scp file for enhancement.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory of GSS enhanced data.",
    )

    return parser


def wfread(f):
    fs, data = wf.read(f)
    if data.dtype == np.int16:
        data = np.float32(data) / 32768
    return data, fs


def wfwrite(z, fs, store_path):
    tmpwav = np.int16(z * 32768)
    wf.write(store_path, fs, tmpwav)


def main():
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    stft_window, stft_shift = 512, 256
    gss = GSS(iterations=20, iterations_post=1)
    bf = Beamformer("mvdrSouden_ban", "mask_mul")

    with codecs.open(args.wav_scp, "r") as handle:
        lines_content = handle.readlines()
    wav_lines = [*map(lambda x: x[:-1] if x[-1] in ["\n"] else x, lines_content)]

    cnt = 0
    
    session2spk2dur = {}
    with codecs.open(args.segments, "r") as handle:
        for line in handle.readlines():
            uttid, spkid, stime, etime = line.strip().split(" ")
            sessionid = spkid.split("-")[0]
            if sessionid not in session2spk2dur.keys():
                session2spk2dur[sessionid] = {}
            if spkid not in session2spk2dur[sessionid].keys():
                session2spk2dur[sessionid][spkid] = []
            session2spk2dur[sessionid][spkid].append((float(stime), float(etime)))
    # import pdb;pdb.set_trace()
    
    for wav_idx in tqdm(range(len(wav_lines)), leave=True, desc="0"):
        # get wav files from scp file
        file_list = wav_lines[wav_idx].split(" ")
        sessionid, wav_list = file_list[0], file_list[1:]

        signal_list = []
        time_activity = []
        cnt += 1
        logging.info("Processing {}: {}".format(cnt, wav_list[0]))

        # read all wavs
        for wav in wav_list:
            data, fs = wfread(wav)
            signal_list.append(data)
        try:
            obstft = np.stack(signal_list, axis=0)
        except:
            minlen = min([len(s) for s in signal_list])
            obstft = np.stack([s[:minlen] for s in signal_list])
        wavlen = obstft.shape[1]
        obstft = stft(obstft, stft_window, stft_shift)

        # get activated timestamps and frequencies
        speaker_list = []
        for spk, dur in session2spk2dur[sessionid].items():
            speaker_list.append(spk.split("-")[-1])
            time_activity.append(get_time_activity(dur, wavlen, fs))
        time_activity.append([True] * wavlen)
        frequency_activity = get_frequency_activity(
            time_activity, stft_window, stft_shift
        )
        # import pdb;pdb.set_trace()

        # generate mask
        masks = gss(obstft, frequency_activity)
        masks_bak = masks

        for i in range(masks.shape[0] - 1):
            target_mask = masks[i]
            distortion_mask = np.sum(np.delete(masks, i, axis=0), axis=0)
            xhat = bf(obstft, target_mask=target_mask, distortion_mask=distortion_mask)
            xhat = istft(xhat, stft_window, stft_shift)
            audio_dir = "/".join(wav_list[0].split("/")[:-1])
            store_path = (
                wav_list[0]
                .replace(audio_dir, args.output_dir)
                .replace(".wav", "-{}.wav".format(speaker_list[i]))
            )
            if not os.path.exists(os.path.split(store_path)[0]):
                os.makedirs(os.path.split(store_path)[0], exist_ok=True)

            logging.info("Save wav file {}.".format(store_path))
            wfwrite(xhat, fs, store_path)
            masks = masks_bak


if __name__ == "__main__":
    main()
