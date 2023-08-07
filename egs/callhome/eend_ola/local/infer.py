import argparse
import os

import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import medfilt

import funasr.models.frontend.eend_ola_feature as eend_ola_feature
from funasr.build_utils.build_model_from_file import build_model_from_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="model config file",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        help="model path",
    )
    parser.add_argument(
        "--output_rttm_file",
        type=str,
        help="output rttm path",
    )
    parser.add_argument(
        "--wav_scp_file",
        type=str,
        default="wav.scp",
        help="input data path",
    )
    parser.add_argument(
        "--frame_shift",
        type=int,
        default=80,
        help="frame shift",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        default=200,
        help="frame size",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=7,
        help="context size",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=8000,
        help="sampling rate",
    )
    parser.add_argument(
        "--subsampling",
        type=int,
        default=10,
        help="setting subsampling",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="shuffle speech in time",
    )
    parser.add_argument(
        "--attractor_threshold",
        type=float,
        default=0.5,
        help="threshold for selecting attractors",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        configs = yaml.safe_load(f)
        for k, v in configs.items():
            if not hasattr(args, k):
                setattr(args, k, v)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTORCH_SEED'] = str(args.seed)

    model, _ = build_model_from_file(config_file=args.config_file, model_file=args.model_file, task_name="diar",
                                  device=args.device)
    model.eval()

    with open(args.wav_scp_file) as f:
        wav_lines = [line.strip().split() for line in f.readlines()]
        wav_items = {x[0]: x[1] for x in wav_lines}

    print("Start inference")
    with open(args.output_rttm_file, "w") as wf:
        for wav_id in wav_items.keys():
            print("Process wav: {}".format(wav_id))
            data, rate = sf.read(wav_items[wav_id])
            speech = eend_ola_feature.stft(data, args.frame_size, args.frame_shift)
            speech = eend_ola_feature.transform(speech)
            speech = eend_ola_feature.splice(speech, context_size=args.context_size)
            speech = speech[::args.subsampling]  # sampling
            speech = torch.from_numpy(speech)

            with torch.no_grad():
                speech = speech.to(args.device)
                ys, _, _, _ = model.estimate_sequential(
                    [speech],
                    n_speakers=None,
                    th=args.attractor_threshold,
                    shuffle=args.shuffle
                )

            a = ys[0].cpu().numpy()
            a = medfilt(a, (11, 1))
            rst = []
            for spkr_id, frames in enumerate(a.T):
                frames = np.pad(frames, (1, 1), 'constant')
                changes, = np.where(np.diff(frames, axis=0) != 0)
                fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
                for s, e in zip(changes[::2], changes[1::2]):
                    st = s * args.frame_shift * args.subsampling / args.sampling_rate
                    dur = (e - s) * args.frame_shift * args.subsampling / args.sampling_rate
                    print(fmt.format(
                        wav_id,
                        st,
                        dur,
                        wav_id + "_" + str(spkr_id)), file=wf)