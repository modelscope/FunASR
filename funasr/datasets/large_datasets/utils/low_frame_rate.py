import numpy as np


def build_LFR_features(data, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.

    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """

    LFR_inputs = []
    T = data.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(data[i*n:i*n+m]))
        else:
            num_padding = m - (T - i * n)
            frame = np.hstack(data[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, data[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)
