#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# This script generates simulated multi-talker mixtures for diarization
#
# common/make_mixture.py \
#     mixture.scp \
#     data/mixture \
#     wav/mixture


import argparse
import os
from eend import kaldi_data
import numpy as np
import math
import soundfile as sf
import json

parser = argparse.ArgumentParser()
parser.add_argument('script',
                    help='list of json')
parser.add_argument('out_data_dir',
                    help='output data dir of mixture')
parser.add_argument('out_wav_dir',
                    help='output mixture wav files are stored here')
parser.add_argument('--rate', type=int, default=16000,
                    help='sampling rate')
args = parser.parse_args()

# open output data files
segments_f = open(args.out_data_dir + '/segments', 'w')
utt2spk_f = open(args.out_data_dir + '/utt2spk', 'w')
wav_scp_f = open(args.out_data_dir + '/wav.scp', 'w')

# "-R" forces the default random seed for reproducibility
resample_cmd = "sox -R -t wav - -t wav - rate {}".format(args.rate)

for line in open(args.script):
    recid, jsonstr = line.strip().split(None, 1)
    indata = json.loads(jsonstr)
    wavfn = indata['recid']
    # recid now include out_wav_dir
    recid = os.path.join(args.out_wav_dir, wavfn).replace('/','_')
    noise = indata['noise']
    noise_snr = indata['snr']
    mixture = []
    for speaker in indata['speakers']:
        spkid = speaker['spkid']
        utts = speaker['utts']
        intervals = speaker['intervals']
        rir = speaker['rir']
        data = []
        pos = 0
        for interval, utt in zip(intervals, utts):
            # append silence interval data
            silence = np.zeros(int(interval * args.rate))
            data.append(silence)
            # utterance is reverberated using room impulse response
            preprocess = "wav-reverberate --print-args=false " \
                         " --impulse-response={} - -".format(rir)
            if isinstance(utt, list):
                rec, st, et = utt
                st = np.rint(st * args.rate).astype(int)
                et = np.rint(et * args.rate).astype(int)
            else:
                rec = utt
                st = 0
                et = None
            if rir is not None:
                wav_rxfilename = kaldi_data.process_wav(rec, preprocess)
            else:
                wav_rxfilename = rec
            wav_rxfilename = kaldi_data.process_wav(
                    wav_rxfilename, resample_cmd)
            speech, _ = kaldi_data.load_wav(wav_rxfilename, st, et)
            data.append(speech)
            # calculate start/end position in samples
            startpos = pos + len(silence)
            endpos = startpos + len(speech)
            # write segments and utt2spk
            uttid = '{}_{}_{:07d}_{:07d}'.format(
                    spkid, recid, int(startpos / args.rate * 100),
                    int(endpos / args.rate * 100))
            print(uttid, recid,
                  startpos / args.rate, endpos / args.rate, file=segments_f)
            print(uttid, spkid, file=utt2spk_f)
            # update position for next utterance
            pos = endpos
        data = np.concatenate(data)
        mixture.append(data)

    # fitting to the maximum-length speaker data, then mix all speakers
    maxlen = max(len(x) for x in mixture)
    mixture = [np.pad(x, (0, maxlen - len(x)), 'constant') for x in mixture]
    mixture = np.sum(mixture, axis=0)
    # noise is repeated or cutted for fitting to the mixture data length
    noise_resampled = kaldi_data.process_wav(noise, resample_cmd)
    noise_data, _ = kaldi_data.load_wav(noise_resampled)
    if maxlen > len(noise_data):
        noise_data = np.pad(noise_data, (0, maxlen - len(noise_data)), 'wrap')
    else:
        noise_data = noise_data[:maxlen]
    # noise power is scaled according to selected SNR, then mixed
    signal_power = np.sum(mixture**2) / len(mixture)
    noise_power = np.sum(noise_data**2) / len(noise_data)
    scale = math.sqrt(
                math.pow(10, - noise_snr / 10) * signal_power / noise_power)
    mixture += noise_data * scale
    # output the wav file and write wav.scp
    outfname = '{}.wav'.format(wavfn)
    outpath = os.path.join(args.out_wav_dir, outfname)
    sf.write(outpath, mixture, args.rate)
    print(recid, os.path.abspath(outpath), file=wav_scp_f)

wav_scp_f.close()
segments_f.close()
utt2spk_f.close()
