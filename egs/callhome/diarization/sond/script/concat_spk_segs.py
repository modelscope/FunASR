import numpy as np
from funasr.utils.job_runner import MultiProcessRunnerV3
from funasr.utils.misc import load_scp_as_list, load_scp_as_dict
import os
import librosa
import soundfile as sf
import argparse


class MyRunner(MultiProcessRunnerV3):

    def prepare(self, parser):
        parser.add_argument("dir", type=str)
        parser.add_argument("out_dir", type=str)
        args = parser.parse_args()
        assert args.sr == 8000, "For callhome dataset, the sample rate should be 8000, use --sr 8000."

        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

        print("loading data...")
        wav_scp = load_scp_as_dict(os.path.join(args.dir, "wav.scp"))
        utt2spk = load_scp_as_dict(os.path.join(args.dir, "utt2spk"))

        spk2utt = {}
        count = 0
        for utt, spk in utt2spk.items():
            if utt in wav_scp:
                if spk not in spk2utt:
                    spk2utt[spk] = []
                spk2utt[spk].append(utt)
                count += 1
        task_list = spk2utt.keys()
        print("total: {} speakers, {} utterances".format(len(spk2utt), count))
        print("starting jobs...")
        return task_list, [spk2utt, wav_scp], args

    def post(self, results_list, args):
        pass


def process(task_args):
    _, task_list, [spk2utt, wav_scp], args = task_args
    for spk in task_list:
        spk_wav_list = []
        for utt in spk2utt[spk]:
            wav = librosa.load(wav_scp[utt], sr=args.sr, mono=True)[0] * 32767
            spk_wav_list.append(wav)
        sig = np.concatenate(spk_wav_list, axis=0)
        save_path = os.path.join(args.out_dir, "{}.wav".format(spk))
        sf.write(save_path, sig.astype(np.int16), args.sr, "PCM_16", "LITTLE", "WAV", True)
    return None


if __name__ == '__main__':
    my_runner = MyRunner(process)
    my_runner.run()
