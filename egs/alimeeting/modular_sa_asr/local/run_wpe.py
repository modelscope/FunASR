#!/usr/bin/env python
# _*_ coding: UTF-8 _*_

import argparse
import codecs
import os
import logging
from multiprocessing import Pool

import numpy as np
import scipy.io.wavfile as wf
from nara_wpe.utils import istft, stft
from nara_wpe.wpe import wpe_v8 as wpe


def wpe_worker(
    wav_scp,
    audio_dir="",
    output_dir="",
    channel=0,
    processing_id=None,
    processing_num=None,
):
    sampling_rate = 16000
    iterations = 5
    stft_options = dict(
        size=512,
        shift=128,
        window_length=None,
        fading=True,
        pad=True,
        symmetric_window=False,
    )
    with codecs.open(wav_scp, "r") as handle:
        lines_content = handle.readlines()
    wav_lines = [*map(lambda x: x[:-1] if x[-1] in ["\n"] else x, lines_content)]
    for wav_idx in range(len(wav_lines)):
        if processing_id is None:
            processing_token = True
        else:
            if wav_idx % processing_num == processing_id:
                processing_token = True
            else:
                processing_token = False
        if processing_token:
            wav_list = wav_lines[wav_idx].split(" ")
            file_exist = True
            for wav_path in wav_list:
                file_exist = file_exist and os.path.exists(
                    wav_path.replace(audio_dir, output_dir)
                )
                if not file_exist:
                    break
            if not file_exist:
                logging.info("wait to process {} : {}".format(wav_idx, wav_list[0]))
                signal_list = []
                for f in wav_list:
                    _, data = wf.read(f)
                    data = data[:, channel - 1]
                    if data.dtype == np.int16:
                        data = np.float32(data) / 32768
                    signal_list.append(data)
                min_len = len(signal_list[0])
                max_len = len(signal_list[0])
                for i in range(1, len(signal_list)):
                    min_len = min(min_len, len(signal_list[i]))
                    max_len = max(max_len, len(signal_list[i]))
                if min_len != max_len:
                    for i in range(len(signal_list)):
                        signal_list[i] = signal_list[i][:min_len]
                y = np.stack(signal_list, axis=0)
                Y = stft(y, **stft_options).transpose(2, 0, 1)
                Z = wpe(Y, iterations=iterations, statistics_mode="full").transpose(
                    1, 2, 0
                )
                z = istft(Z, size=stft_options["size"], shift=stft_options["shift"])
                for d in range(len(signal_list)):
                    store_path = wav_list[d].replace(audio_dir, output_dir)
                    if not os.path.exists(os.path.split(store_path)[0]):
                        os.makedirs(os.path.split(store_path)[0], exist_ok=True)
                    tmpwav = np.int16(z[d, :] * 32768)
                    wf.write(store_path, sampling_rate, tmpwav)
            else:
                logging.info("file exist {} : {}".format(wav_idx, wav_list[0]))
    return None


def wpe_manager(
    wav_scp, processing_num=1, audio_dir="", output_dir="", channel=1
):
    if processing_num > 1:
        pool = Pool(processes=processing_num)
        for i in range(processing_num):
            pool.apply_async(
                wpe_worker,
                kwds={
                    "wav_scp": wav_scp,
                    "processing_id": i,
                    "processing_num": processing_num,
                    "audio_dir": audio_dir,
                    "output_dir": output_dir,
                },
            )
        pool.close()
        pool.join()
    else:
        wpe_worker(wav_scp, audio_dir=audio_dir, output_dir=output_dir, channel=channel)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_wpe")
    parser.add_argument(
        "--wav-scp",
        type=str,
        required=True,
        help="Path pf wav scp file",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        required=True,
        help="Directory of input audio files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory of WPE enhanced audio files",
    )
    parser.add_argument(
        "--channel",
        type=str,
        required=True,
        help="Channel number of input audio",
    )
    parser.add_argument("--nj", type=int, default="1", help="number of process")
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    logging.info("wavfile={}".format(args.wav_scp))
    logging.info("processingnum={}".format(args.nj))

    wpe_manager(
        wav_scp=args.wav_scp,
        processing_num=args.nj,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        channel=int(args.channel)
    )
