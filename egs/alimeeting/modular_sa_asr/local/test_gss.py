import io
import functools
import logging

# import soundfile as sf
import numpy as np
import matplotlib
import matplotlib.pylab as plt

# from IPython.display import display, Audio

from nara_wpe.utils import stft, istft

from pb_bss.distribution import CACGMMTrainer
from pb_bss.evaluation import InputMetrics, OutputMetrics
from dataclasses import dataclass

# from beamforming_wrapper import beamform_mvdr_souden_from_masks
from pb_chime5.utils.numpy_utils import segment_axis_v2
from textgrid_processor import read_textgrid_from_file


def get_time_activity(dur_list, wavlen, sr):
    time_activity = [False] * wavlen
    for dur in dur_list:
        xmax = int(dur[1] * sr)
        xmin = int(dur[0] * sr)
        if xmax > wavlen:
            continue
        for i in range(xmin, xmax):
            time_activity[i] = True
    logging.info("Num of actived samples {}".format(time_activity.count(True)))
    return time_activity

def get_frequency_activity(
    time_activity,
    stft_window_length,
    stft_shift,
    stft_fading=True,
    stft_pad=True,
):
    time_activity = np.asarray(time_activity)

    if stft_fading:
        pad_width = np.array([(0, 0)] * time_activity.ndim)
        pad_width[-1, :] = stft_window_length - stft_shift  # Consider fading
        time_activity = np.pad(time_activity, pad_width, mode="constant")

    return segment_axis_v2(
        time_activity,
        length=stft_window_length,
        shift=stft_shift,
        end="pad" if stft_pad else "cut",
    ).any(axis=-1)


@dataclass
class Beamformer:
    type: str
    postfilter: str

    def __call__(self, Obs, target_mask, distortion_mask, debug=False):
        bf = self.type

        if bf == "mvdrSouden_ban":
            from pb_chime5.speech_enhancement.beamforming_wrapper import (
                beamform_mvdr_souden_from_masks,
            )

            X_hat = beamform_mvdr_souden_from_masks(
                Y=Obs,
                X_mask=target_mask,
                N_mask=distortion_mask,
                ban=True,
            )
        elif bf == "ch0":
            X_hat = Obs[0]
        elif bf == "sum":
            X_hat = np.sum(Obs, axis=0)
        else:
            raise NotImplementedError(bf)

        if self.postfilter is None:
            pass
        elif self.postfilter == "mask_mul":
            X_hat = X_hat * target_mask
        else:
            raise NotImplementedError(self.postfilter)

        return X_hat


@dataclass
class GSS:
    iterations: int = 20
    iterations_post: int = 0
    verbose: bool = True
    # use_pinv: bool = False
    # stable: bool = True

    def __call__(self, Obs, acitivity_freq=None, debug=False):
        initialization = np.asarray(acitivity_freq, dtype=np.float64)
        initialization = np.where(initialization == 0, 1e-10, initialization)
        initialization = initialization / np.sum(initialization, keepdims=True, axis=0)
        initialization = np.repeat(initialization[None, ...], 257, axis=0)

        source_active_mask = np.asarray(acitivity_freq, dtype=bool)
        source_active_mask = np.repeat(source_active_mask[None, ...], 257, axis=0)

        cacGMM = CACGMMTrainer()

        if debug:
            learned = []
        all_affiliations = []
        F = Obs.shape[-1]
        T = Obs.T.shape[-2]

        for f in range(F):
            if self.verbose:
                if f % 50 == 0:
                    logging.info(f"{f}/{F}")

            # T: Consider end of signal.
            # This should not be nessesary, but activity is for inear and not for
            # array.
            cur = cacGMM.fit(
                y=Obs.T[f, ...],
                initialization=initialization[f, ..., :T],
                iterations=self.iterations,
                source_activity_mask=source_active_mask[f, ..., :T],
            )
            affiliation = cur.predict(
                Obs.T[f, ...],
                source_activity_mask=source_active_mask[f, ..., :T],
            )

            all_affiliations.append(affiliation)

        posterior = np.array(all_affiliations).transpose(1, 2, 0)

        return posterior
