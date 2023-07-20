import argparse
import os

import numpy as np

import funasr.modules.eend_ola.utils.feature as feature
import funasr.modules.eend_ola.utils.kaldi_data as kaldi_data


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


class KaldiDiarizationDataset():
    def __init__(
            self,
            data_dir,
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
        self.chunk_size = chunk_size
        self.context_size = context_size
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.chunk_indices = []
        self.label_delay = label_delay

        self.data = kaldi_data.KaldiData(self.data_dir)

        # make chunk indices: filepath, start_frame, end_frame
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
    f = open(out_wav_file, 'w')
    dataset = KaldiDiarizationDataset(
        data_dir=args.data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=8000,
        use_last_samples=True,
    )
    length = len(dataset.chunk_indices)
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
        suffix = '_' + st + '_' + ed

        parts = os.readlink('/'.join(path.split('/')[:-1])).split('/')
        # print('parts: ', parts)
        parts = parts[:4] + ['numpy_data'] + parts[4:]
        cur_path = '/'.join(parts)
        # print('cur path: ', cur_path)
        out_path = os.path.join(cur_path, path.split('/')[-1].split('.')[0] + suffix + '.npz')
        # print(out_path)
        # print(cur_path)
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)
        np.savez(out_path, Y=Y_ss, T=T_ss)
        if idx == length - 1:
            f.write(rec + suffix + ' ' + out_path)
        else:
            f.write(rec + suffix + ' ' + out_path + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("num_frames")
    parser.add_argument("context_size")
    parser.add_argument("frame_size")
    parser.add_argument("frame_shift")
    parser.add_argument("subsampling")



    args = parser.parse_args()
    convert(args)
