import numpy as np

from .feature import sliding_window


# ---------- compute-vad ----------

def compute_vad(log_energy, energy_mean_scale=0.5, energy_threshold=0.5, frames_context=0, proportion_threshold=0.6):
    """ Apply voice activity detection

    :param log_energy: Log mel energy.
    :param energy_mean_scale: If this is set to s, to get the actual threshold we let m be the mean log-energy of the file, and use s*m + vad-energy-threshold (float, default = 0.5)
    :param energy_threshold: Constant term in energy threshold for VAD (also see energy_mean_scale) (float, default = 5)
    :param frames_context: Number of frames of context on each side of central frame, in window for which energy is monitored (int, default = 0)
    :param proportion_threshold: Parameter controlling the proportion of frames within the window that need to have more energy than the threshold (float, default = 0.6)
    :return: A vector of boolean that are True if we judge the frame voiced and False otherwise.
    """
    assert len(log_energy.shape) == 1
    assert energy_mean_scale >= 0
    assert frames_context >= 0
    assert 0 < proportion_threshold < 1
    dtype = log_energy.dtype
    energy_threshold += energy_mean_scale * log_energy.mean()
    if frames_context > 0:
        num_frames = len(log_energy)
        window_size = frames_context * 2 + 1
        log_energy_pad = np.concatenate([
            np.zeros(frames_context, dtype=dtype),
            log_energy,
            np.zeros(frames_context, dtype=dtype)
        ])
        log_energy_window = sliding_window(log_energy_pad, window_size, 1)
        num_count = np.count_nonzero(log_energy_window > energy_threshold, axis=1)
        den_count = np.ones(num_frames, dtype=dtype) * window_size
        max_den_count = np.arange(frames_context + 1, min(window_size, num_frames) + 1, dtype=dtype)
        den_count[:-(frames_context + 2):-1] = max_den_count
        den_count[:frames_context + 1] = np.min([den_count[:frames_context + 1], max_den_count], axis=0)
        vad = num_count / den_count >= proportion_threshold
    else:
        vad = log_energy > energy_threshold
    return vad

# ---------- compute-vad ----------
