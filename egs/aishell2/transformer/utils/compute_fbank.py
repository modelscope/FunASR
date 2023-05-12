from kaldiio import WriteHelper

import argparse
import numpy as np
import json
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


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


def get_parser():
    parser = argparse.ArgumentParser(
        description="computer features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wav-lists",
        "-w",
        default=False,
        required=True,
        type=str,
        help="input wav lists",
    )
    parser.add_argument(
        "--text-files",
        "-t",
        default=False,
        required=True,
        type=str,
        help="input text files",
    )
    parser.add_argument(
        "--dims",
        "-d",
        default=80,
        type=int,
        help="feature dims",
    )
    parser.add_argument(
        "--max-lengths",
        "-m",
        default=1500,
        type=int,
        help="max frame numbers",
    )
    parser.add_argument(
        "--sample-frequency",
        "-s",
        default=16000,
        type=int,
        help="sample frequency",
    )
    parser.add_argument(
        "--speed-perturb",
        "-p",
        default="1.0",
        type=str,
        help="speed perturb",
    )
    parser.add_argument(
        "--ark-index",
        "-a",
        default=1,
        required=True,
        type=int,
        help="ark index",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=False,
        required=True,
        type=str,
        help="output dir",
    )
    parser.add_argument(
        "--window-type",
        default="hamming",
        required=False,
        type=str,
        help="window type"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    ark_file = args.output_dir + "/ark/feats." + str(args.ark_index) + ".ark"
    scp_file = args.output_dir + "/ark/feats." + str(args.ark_index) + ".scp"
    text_file = args.output_dir + "/txt/text." + str(args.ark_index) + ".txt"  
    feats_shape_file = args.output_dir + "/ark/len." + str(args.ark_index)
    text_shape_file = args.output_dir + "/txt/len." + str(args.ark_index)

    ark_writer = WriteHelper('ark,scp:{},{}'.format(ark_file, scp_file))
    text_writer = open(text_file, 'w')
    feats_shape_writer = open(feats_shape_file, 'w')
    text_shape_writer = open(text_shape_file, 'w')

    speed_perturb_list = args.speed_perturb.split(',')
    
    for speed in speed_perturb_list:
        with open(args.wav_lists, 'r', encoding='utf-8') as wavfile:
            with open(args.text_files, 'r', encoding='utf-8') as textfile:
                for wav, text in zip(wavfile, textfile): 
                    s_w = wav.strip().split()
                    wav_id = s_w[0]
                    wav_file = s_w[1]

                    s_t = text.strip().split()
                    text_id = s_t[0]
                    txt = s_t[1:]
                    fbank = compute_fbank(wav_file,
                                          num_mel_bins=args.dims,
                                          resample_rate=args.sample_frequency,
                                          speed=float(speed),
                                          window_type=args.window_type
                                          )
                    feats_dims = fbank.shape[1]
                    feats_lens = fbank.shape[0]
                    if feats_lens >= args.max_lengths:
                        continue
                    txt_lens = len(txt)
                    if speed == "1.0":
                        wav_id_sp = wav_id
                    else: 
                        wav_id_sp = wav_id + "_sp" + speed

                    feats_shape_writer.write(wav_id_sp + " " + str(feats_lens) + "," + str(feats_dims) + '\n')
                    text_shape_writer.write(wav_id_sp + " " + str(txt_lens) + '\n')

                    text_writer.write(wav_id_sp + " " + " ".join(txt) + '\n')
                    ark_writer(wav_id_sp, fbank)
                    

if __name__ == '__main__':
    main()

