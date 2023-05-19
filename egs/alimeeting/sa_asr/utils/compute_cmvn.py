import argparse
import json
import os

import numpy as np
import torchaudio
import torchaudio.compliance.kaldi as kaldi


def get_parser():
    parser = argparse.ArgumentParser(
        description="computer global cmvn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dim",
        default=80,
        type=int,
        help="feature dimension",
    )
    parser.add_argument(
        "--wav_path",
        default=False,
        required=True,
        type=str,
        help="the path of wav scps",
    )
    parser.add_argument(
        "--idx",
        default=1,
        required=True,
        type=int,
        help="index",
    )
    return parser


def compute_fbank(wav_file,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0,
                  resample_rate=16000,
                  speed=1.0,
                  window_type="hamming"):
    waveform, sample_rate = torchaudio.load(wav_file)
    if resample_rate != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                  new_freq=resample_rate)(waveform)
    if speed != 1.0:
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, resample_rate,
            [['speed', str(speed)], ['rate', str(resample_rate)]]
        )

    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      window_type=window_type,
                      sample_frequency=resample_rate)

    return mat.numpy()


def main():
    parser = get_parser()
    args = parser.parse_args()

    wav_scp_file = os.path.join(args.wav_path, "wav.{}.scp".format(args.idx))
    cmvn_file = os.path.join(args.wav_path, "cmvn.{}.json".format(args.idx))

    mean_stats = np.zeros(args.dim)
    var_stats = np.zeros(args.dim)
    total_frames = 0

    # with ReadHelper('ark:{}'.format(ark_file)) as ark_reader:
    #     for key, mat in ark_reader:
    #         mean_stats += np.sum(mat, axis=0)
    #         var_stats += np.sum(np.square(mat), axis=0)
    #         total_frames += mat.shape[0]
    with open(wav_scp_file) as f:
        lines = f.readlines()
        for line in lines:
            _, wav_file = line.strip().split()
            fbank = compute_fbank(wav_file, num_mel_bins=args.dim)
            mean_stats += np.sum(fbank, axis=0)
            var_stats += np.sum(np.square(fbank), axis=0)
            total_frames += fbank.shape[0]

    cmvn_info = {
        'mean_stats': list(mean_stats.tolist()),
        'var_stats': list(var_stats.tolist()),
        'total_frames': total_frames
    }
    with open(cmvn_file, 'w') as fout:
        fout.write(json.dumps(cmvn_info))


if __name__ == '__main__':
    main()
