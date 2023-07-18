#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

"""
This script generates random multi-talker mixtures for diarization.
It generates a scp-like outputs: lines of "[recid] [json]".
    recid: recording id of mixture
        serial numbers like mix_0000001, mix_0000002, ...
    json: mixture configuration formatted in "one-line"
The json format is as following:
{
 'speakers':[                    # list of speakers
    {
     'spkid': 'Name',             # speaker id
     'rir': '/rirdir/rir.wav',    # wav_rxfilename of room impulse response
     'utts': [                    # list of wav_rxfilenames of utterances
        '/wavdir/utt1.wav',
        '/wavdir/utt2.wav',...],
     'intervals': [1.2, 3.4, ...] # list of silence durations before utterances
    }, ... ],
 'noise': '/noisedir/noise.wav'   # wav_rxfilename of background noise
 'snr': 15.0,                     # SNR for mixing background noise
 'recid': 'mix_000001'            # recording id of the mixture
}

Usage:
    common/random_mixture.py \
        --n_mixtures=10000 \      # number of mixtures
        data/voxceleb1_train \    # kaldi-style data dir of utterances
        data/musan_noise_bg \     # background noises
        data/simu_rirs \          # room impulse responses
        > mixture.scp             # output scp-like file

The actual data dir and wav files are generated using make_mixture.py:
    common/make_mixture.py \
        mixture.scp \             # scp-like file for mixture
        data/mixture \            # output data dir
        wav/mixture               # output wav dir
"""

import argparse
import os
from eend import kaldi_data
import random
import numpy as np
import json
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('data_dir',
                    help='data dir of single-speaker recordings')
parser.add_argument('noise_dir',
                    help='data dir of background noise recordings')
parser.add_argument('rir_dir',
                    help='data dir of room impulse responses')
parser.add_argument('--n_mixtures', type=int, default=10,
                    help='number of mixture recordings')
parser.add_argument('--n_speakers', type=int, default=4,
                    help='number of speakers in a mixture')
parser.add_argument('--min_utts', type=int, default=10,
                    help='minimum number of uttenraces per speaker')
parser.add_argument('--max_utts', type=int, default=20,
                    help='maximum number of utterances per speaker')
parser.add_argument('--sil_scale', type=float, default=10.0,
                    help='average silence time')
parser.add_argument('--noise_snrs', default="10:15:20",
                    help='colon-delimited SNRs for background noises')
parser.add_argument('--random_seed', type=int, default=777,
                    help='random seed')
parser.add_argument('--speech_rvb_probability', type=float, default=1,
                    help='reverb probability')
args = parser.parse_args()

random.seed(args.random_seed)
np.random.seed(args.random_seed)

# load list of wav files from kaldi-style data dirs
wavs = kaldi_data.load_wav_scp(
        os.path.join(args.data_dir, 'wav.scp'))
noises = kaldi_data.load_wav_scp(
        os.path.join(args.noise_dir, 'wav.scp'))
rirs = kaldi_data.load_wav_scp(
        os.path.join(args.rir_dir, 'wav.scp'))

# spk2utt is used for counting number of utterances per speaker
spk2utt = kaldi_data.load_spk2utt(
        os.path.join(args.data_dir, 'spk2utt'))

segments = kaldi_data.load_segments_hash(
        os.path.join(args.data_dir, 'segments'))

# choice lists for random sampling
all_speakers = list(spk2utt.keys())
all_noises = list(noises.keys())
all_rirs = list(rirs.keys())
noise_snrs = [float(x) for x in args.noise_snrs.split(':')]

mixtures = []
for it in range(args.n_mixtures):
    # recording ids are mix_0000001, mix_0000002, ...
    recid = 'mix_{:07d}'.format(it + 1)
    # randomly select speakers, a background noise and a SNR
    speakers = random.sample(all_speakers, args.n_speakers)
    noise = random.choice(all_noises)
    noise_snr = random.choice(noise_snrs)
    mixture = {'speakers': []}
    for speaker in speakers:
        # randomly select the number of utterances
        n_utts = np.random.randint(args.min_utts, args.max_utts + 1)
        # utts = spk2utt[speaker][:n_utts]
        cycle_utts = itertools.cycle(spk2utt[speaker])
        # random start utterance
        roll = np.random.randint(0, len(spk2utt[speaker]))
        for i in range(roll):
            next(cycle_utts)
        utts = [next(cycle_utts) for i in range(n_utts)]
        # randomly select wait time before appending utterance
        intervals = np.random.exponential(args.sil_scale, size=n_utts)
        # randomly select a room impulse response
        if random.random() < args.speech_rvb_probability:
            rir = rirs[random.choice(all_rirs)]
        else:
            rir = None
        if segments is not None:
            utts = [segments[utt] for utt in utts]
            utts = [(wavs[rec], st, et) for (rec, st, et) in utts]
            mixture['speakers'].append({
                'spkid': speaker,
                'rir': rir,
                'utts': utts,
                'intervals': intervals.tolist()
                })
        else:
            mixture['speakers'].append({
                'spkid': speaker,
                'rir': rir,
                'utts': [wavs[utt] for utt in utts],
                'intervals': intervals.tolist()
                })
    mixture['noise'] = noises[noise]
    mixture['snr'] = noise_snr
    mixture['recid'] = recid
    print(recid, json.dumps(mixture))
