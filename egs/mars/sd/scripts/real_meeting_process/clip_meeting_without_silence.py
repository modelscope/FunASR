import numpy as np
from funasr.utils.job_runner import MultiProcessRunnerV3
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse


class MyRunner(MultiProcessRunnerV3):

    def prepare(self, parser):
        parser.add_argument("wav_scp", type=str)
        parser.add_argument("out_dir", type=str)
        parser.add_argument("--chunk_dur", type=float, default=16)
        parser.add_argument("--shift_dur", type=float, default=4)
        args = parser.parse_args()

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        wav_scp = load_scp_as_list(args.wav_scp)
        return wav_scp, None, args

    def post(self, results_list, args):
        pass


def process(task_args):
    _, task_list, _, args = task_args
    chunk_len, shift_len = int(args.chunk_dur * args.sr), int(args.shift_dur * args.sr)
    for mid, wav_path in tqdm(task_list, total=len(task_list), ascii=True, disable=args.no_pbar):
        if not os.path.exists(os.path.join(args.out_dir, mid)):
            os.makedirs(os.path.join(args.out_dir, mid))

        wav = librosa.load(wav_path, args.sr, True)[0] * 32767
        n_chunk = (len(wav) - chunk_len) // shift_len + 1
        if (len(wav) - chunk_len) % shift_len > 0:
            n_chunk += 1
        for i in range(n_chunk):
            seg = wav[i*shift_len: i*shift_len + chunk_len]
            st = int(float(i*shift_len)/args.sr * 100)
            dur = int(float(len(seg))/args.sr * 100)
            file_name = "{}_S{:04d}_{:07d}_{:07d}.wav".format(mid, i, st, st+dur)
            save_path = os.path.join(args.out_dir, mid, file_name)
            sf.write(save_path, seg.astype(np.int16), args.sr, "PCM_16", "LITTLE", "WAV", True)
    return None


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
