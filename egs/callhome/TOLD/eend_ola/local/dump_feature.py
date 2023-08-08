import argparse
import os

from kaldiio import WriteHelper

import funasr.modules.eend_ola.utils.feature as feature
from funasr.modules.eend_ola.utils.kaldi_data import load_segments_rechash, load_utt2spk, load_wav_scp, load_reco2dur, \
    load_spk2utt, load_wav


def _count_frames(data_len, size, step):
    return int((data_len - size + step) / step)


def _gen_frame_indices(
        data_length, size=2000, step=2000,
        use_last_samples=False,
        label_delay=0,
        subsampling=1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i * step, i * step + size
    if use_last_samples and i * step + size < data_length:
        if data_length - (i + 1) * step - subsampling * label_delay > 0:
            yield (i + 1) * step, data_length


class KaldiData:
    def __init__(self, data_dir, idx):
        self.data_dir = data_dir
        segment_file = os.path.join(self.data_dir, 'segments.{}'.format(idx))
        self.segments = load_segments_rechash(segment_file)

        utt2spk_file = os.path.join(self.data_dir, 'utt2spk.{}'.format(idx))
        self.utt2spk = load_utt2spk(utt2spk_file)

        wav_file = os.path.join(self.data_dir, 'wav.scp.{}'.format(idx))
        self.wavs = load_wav_scp(wav_file)

        reco2dur_file = os.path.join(self.data_dir, 'reco2dur.{}'.format(idx))
        self.reco2dur = load_reco2dur(reco2dur_file)

        spk2utt_file = os.path.join(self.data_dir, 'spk2utt.{}'.format(idx))
        self.spk2utt = load_spk2utt(spk2utt_file)

    def load_wav(self, recid, start=0, end=None):
        data, rate = load_wav(self.wavs[recid], start, end)
        return data, rate


class KaldiDiarizationDataset():
    def __init__(
            self,
            data_dir,
            index,
            chunk_size=2000,
            context_size=0,
            frame_size=1024,
            frame_shift=256,
            subsampling=1,
            rate=16000,
            input_transform=None,
            use_last_samples=False,
            label_delay=0,
            n_speakers=None,
    ):
        self.data_dir = data_dir
        self.index = index
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.chunk_indices = []
        self.label_delay = label_delay

        self.data = KaldiData(self.data_dir, index)

        for rec, path in self.data.wavs.items():
            data_len = int(self.data.reco2dur[rec] * rate / frame_shift)
            data_len = int(data_len / self.subsampling)
            for st, ed in _gen_frame_indices(
                    data_len, chunk_size, chunk_size, use_last_samples,
                    label_delay=self.label_delay,
                    subsampling=self.subsampling):
                self.chunk_indices.append(
                    (rec, path, st * self.subsampling, ed * self.subsampling))
        print(len(self.chunk_indices), " chunks")


def convert(args):
    dataset = KaldiDiarizationDataset(
        data_dir=args.data_dir,
        index=args.index,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform="logmel23_mn",
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=8000,
        use_last_samples=True,
    )

    feature_ark_file = os.path.join(args.output_dir, "feature.ark.{}".format(args.index))
    feature_scp_file = os.path.join(args.output_dir, "feature.scp.{}".format(args.index))
    label_ark_file = os.path.join(args.output_dir, "label.ark.{}".format(args.index))
    label_scp_file = os.path.join(args.output_dir, "label.scp.{}".format(args.index))
    with WriteHelper('ark,scp:{},{}'.format(feature_ark_file, feature_scp_file)) as feature_writer, \
            WriteHelper('ark,scp:{},{}'.format(label_ark_file, label_scp_file)) as label_writer:
        for idx, (rec, path, st, ed) in enumerate(dataset.chunk_indices):
            Y, T = feature.get_labeledSTFT(
                dataset.data,
                rec,
                st,
                ed,
                dataset.frame_size,
                dataset.frame_shift,
                dataset.n_speakers)
            Y = feature.transform(Y, dataset.input_transform)
            Y_spliced = feature.splice(Y, dataset.context_size)
            Y_ss, T_ss = feature.subsample(Y_spliced, T, dataset.subsampling)
            st = '{:0>7d}'.format(st)
            ed = '{:0>7d}'.format(ed)
            key = "{}_{}_{}".format(rec, st, ed)
            feature_writer(key, Y_ss)
            label_writer(key, T_ss.reshape(-1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--index", type=str)
    parser.add_argument("--num_frames", type=int, default=500)
    parser.add_argument("--context_size", type=int, default=7)
    parser.add_argument("--frame_size", type=int, default=200)
    parser.add_argument("--frame_shift", type=int, default=80)
    parser.add_argument("--subsampling", type=int, default=10)

    args = parser.parse_args()
    convert(args)
