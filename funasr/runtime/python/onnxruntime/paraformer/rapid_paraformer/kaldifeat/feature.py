import numpy as np
from scipy.fftpack import dct


# ---------- feature-window ----------

def sliding_window(x, window_size, window_shift):
    shape = x.shape[:-1] + (x.shape[-1] - window_size + 1, window_size)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)[::window_shift]


def func_num_frames(num_samples, window_size, window_shift, snip_edges):
    if snip_edges:
        if num_samples < window_size:
            return 0
        else:
            return 1 + ((num_samples - window_size) // window_shift)
    else:
        return (num_samples + (window_shift // 2)) // window_shift


def func_dither(waveform, dither_value):
    if dither_value == 0.0:
        return waveform
    waveform += np.random.normal(size=waveform.shape).astype(waveform.dtype) * dither_value
    return waveform


def func_remove_dc_offset(waveform):
    return waveform - np.mean(waveform)


def func_log_energy(waveform):
    return np.log(np.dot(waveform, waveform).clip(min=np.finfo(waveform.dtype).eps))


def func_preemphasis(waveform, preemph_coeff):
    if preemph_coeff == 0.0:
        return waveform
    assert 0 < preemph_coeff <= 1
    waveform[1:] -= preemph_coeff * waveform[:-1]
    waveform[0] -= preemph_coeff * waveform[0]
    return waveform


def sine(M):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, float)
    n = np.arange(0, M)
    return np.sin(np.pi*n/(M-1))


def povey(M):
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, float)
    n = np.arange(0, M)
    return (0.5 - 0.5*np.cos(2.0*np.pi*n/(M-1)))**0.85


def feature_window_function(window_type, window_size, blackman_coeff):
    assert window_size > 0
    if window_type == 'hanning':
        return np.hanning(window_size)
    elif window_type == 'sine':
        return sine(window_size)
    elif window_type == 'hamming':
        return np.hamming(window_size)
    elif window_type == 'povey':
        return povey(window_size)
    elif window_type == 'rectangular':
        return np.ones(window_size)
    elif window_type == 'blackman':
        window_func = np.blackman(window_size)
        if blackman_coeff == 0.42:
            return window_func
        else:
            return window_func - 0.42 + blackman_coeff
    else:
        raise ValueError('Invalid window type {}'.format(window_type))


def process_window(window, dither, remove_dc_offset, preemphasis_coefficient, window_function, raw_energy):
    if dither != 0.0:
        window = func_dither(window, dither)
    if remove_dc_offset:
        window = func_remove_dc_offset(window)
    if raw_energy:
        log_energy = func_log_energy(window)
    if preemphasis_coefficient != 0.0:
        window = func_preemphasis(window, preemphasis_coefficient)
    window *= window_function
    if not raw_energy:
        log_energy = func_log_energy(window)
    return window, log_energy


def extract_window(waveform, blackman_coeff, dither, window_size, window_shift,
                   preemphasis_coefficient, raw_energy, remove_dc_offset,
                   snip_edges, window_type, dtype):
    num_samples = len(waveform)
    num_frames = func_num_frames(num_samples, window_size, window_shift, snip_edges)
    num_samples_ = (num_frames - 1) * window_shift + window_size
    if snip_edges:
        waveform = waveform[:num_samples_]
    else:
        offset = window_shift // 2 - window_size // 2
        waveform = np.concatenate([
            waveform[-offset - 1::-1],
            waveform,
            waveform[:-(offset + num_samples_ - num_samples + 1):-1]
        ])
    frames = sliding_window(waveform, window_size=window_size, window_shift=window_shift)
    frames = frames.astype(dtype)
    log_enery = np.empty(frames.shape[0], dtype=dtype)
    for i in range(frames.shape[0]):
        frames[i], log_enery[i] = process_window(
            window=frames[i],
            dither=dither,
            remove_dc_offset=remove_dc_offset,
            preemphasis_coefficient=preemphasis_coefficient,
            window_function=feature_window_function(
                window_type=window_type,
                window_size=window_size,
                blackman_coeff=blackman_coeff
            ).astype(dtype),
            raw_energy=raw_energy
        )
    return frames, log_enery

# ---------- feature-window ----------


# ---------- feature-functions ----------

def compute_spectrum(frames, n):
    complex_spec = np.fft.rfft(frames, n)
    return np.absolute(complex_spec)


def compute_power_spectrum(frames, n):
    return np.square(compute_spectrum(frames, n))


def apply_cmvn_sliding_internal(feat, center=False, window=600, min_window=100, norm_vars=False):
    num_frames, feat_dim = feat.shape
    std = 1
    if center:
        if num_frames <= window:
            mean = feat.mean(axis=0, keepdims=True).repeat(num_frames, axis=0)
            if norm_vars:
                std = feat.std(axis=0, keepdims=True).repeat(num_frames, axis=0)
        else:
            feat1 = feat[:window]
            feat2 = sliding_window(feat.T, window, 1)
            feat3 = feat[-window:]
            mean1 = feat1.mean(axis=0, keepdims=True).repeat(window // 2, axis=0)
            mean2 = feat2.mean(axis=2).T
            mean3 = feat3.mean(axis=0, keepdims=True).repeat((window - 1) // 2, axis=0)
            mean = np.concatenate([mean1, mean2, mean3])
            if norm_vars:
                std1 = feat1.std(axis=0, keepdims=True).repeat(window // 2, axis=0)
                std2 = feat2.std(axis=2).T
                std3 = feat3.mean(axis=0, keepdims=True).repeat((window - 1) // 2, axis=0)
                std = np.concatenate([std1, std2, std3])
    else:
        if num_frames <= min_window:
            mean = feat.mean(axis=0, keepdims=True).repeat(num_frames, axis=0)
            if norm_vars:
                std = feat.std(axis=0, keepdims=True).repeat(num_frames, axis=0)
        else:
            feat1 = feat[:min_window]
            mean1 = feat1.mean(axis=0, keepdims=True).repeat(min_window, axis=0)
            feat2_cumsum = np.cumsum(feat[:window], axis=0)[min_window:]
            cumcnt = np.arange(min_window + 1, min(window, num_frames) + 1, dtype=feat.dtype)[:, np.newaxis]
            mean2 = feat2_cumsum / cumcnt
            mean = np.concatenate([mean1, mean2])
            if norm_vars:
                std1 = feat1.std(axis=0, keepdims=True).repeat(min_window, axis=0)
                feat2_power_cumsum = np.cumsum(np.square(feat[:window]), axis=0)[min_window:]
                std2 = np.sqrt(feat2_power_cumsum / cumcnt - np.square(mean2))
                std = np.concatenate([std1, std2])
            if num_frames > window:
                feat3 = sliding_window(feat.T, window, 1)
                mean3 = feat3.mean(axis=2).T
                mean = np.concatenate([mean, mean3[1:]])
                if norm_vars:
                    std3 = feat3.std(axis=2).T
                    std = np.concatenate([std, std3[1:]])
    feat = (feat - mean) / std
    return feat

# ---------- feature-functions ----------


# ---------- mel-computations ----------

def inverse_mel_scale(mel_freq):
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)


def mel_scale(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)


def compute_mel_banks(num_bins, sample_frequency, low_freq, high_freq, n):
    """ Compute Mel banks.

    :param num_bins: Number of triangular mel-frequency bins
    :param sample_frequency: Waveform data sample frequency
    :param low_freq: Low cutoff frequency for mel bins
    :param high_freq: High cutoff frequency for mel bins (if <= 0, offset from Nyquist)
    :param n: Window size
    :return: Mel banks.
    """
    assert num_bins >= 3, 'Must have at least 3 mel bins'
    num_fft_bins = n // 2

    nyquist = 0.5 * sample_frequency
    if high_freq <= 0:
        high_freq = nyquist + high_freq
    assert 0 <= low_freq < high_freq <= nyquist

    fft_bin_width = sample_frequency / n

    mel_low_freq = mel_scale(low_freq)
    mel_high_freq = mel_scale(high_freq)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    mel_banks = np.zeros([num_bins, num_fft_bins + 1])
    for i in range(num_bins):
        left_mel = mel_low_freq + mel_freq_delta * i
        center_mel = left_mel + mel_freq_delta
        right_mel = center_mel + mel_freq_delta
        for j in range(num_fft_bins):
            mel = mel_scale(fft_bin_width * j)
            if left_mel < mel < right_mel:
                if mel <= center_mel:
                    mel_banks[i, j] = (mel - left_mel) / (center_mel - left_mel)
                else:
                    mel_banks[i, j] = (right_mel - mel) / (right_mel - center_mel)
    return mel_banks


def compute_lifter_coeffs(q, M):
    """ Compute liftering coefficients (scaling on cepstral coeffs)
        the zeroth index is C0, which is not affected.

    :param q: Number of lifters
    :param M: Number of coefficients
    :return: Lifters.
    """
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, float)
    n = np.arange(0, M)
    return 1 + 0.5*np.sin(np.pi*n/q)*q

# ---------- mel-computations ----------


# ---------- compute-fbank-feats ----------

def compute_fbank_feats(
        waveform,
        blackman_coeff=0.42,
        dither=1.0,
        energy_floor=1.0,
        frame_length=25,
        frame_shift=10,
        high_freq=0,
        low_freq=20,
        num_mel_bins=23,
        preemphasis_coefficient=0.97,
        raw_energy=True,
        remove_dc_offset=True,
        round_to_power_of_two=True,
        sample_frequency=16000,
        snip_edges=True,
        use_energy=False,
        use_log_fbank=True,
        use_power=True,
        window_type='povey',
        dtype=np.float32):
    """ Compute (log) Mel filter bank energies

    :param waveform: Input waveform.
    :param blackman_coeff: Constant coefficient for generalized Blackman window. (float, default = 0.42)
    :param dither: Dithering constant (0.0 means no dither). If you turn this off, you should set the --energy-floor option, e.g. to 1.0 or 0.1 (float, default = 1)
    :param energy_floor: Floor on energy (absolute, not relative) in FBANK computation. Only makes a difference if --use-energy=true; only necessary if --dither=0.0.  Suggested values: 0.1 or 1.0 (float, default = 0)
    :param frame_length: Frame length in milliseconds (float, default = 25)
    :param frame_shift: Frame shift in milliseconds (float, default = 10)
    :param high_freq: High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)
    :param low_freq: Low cutoff frequency for mel bins (float, default = 20)
    :param num_mel_bins: Number of triangular mel-frequency bins (int, default = 23)
    :param preemphasis_coefficient: Coefficient for use in signal preemphasis (float, default = 0.97)
    :param raw_energy: If true, compute energy before preemphasis and windowing (bool, default = true)
    :param remove_dc_offset: Subtract mean from waveform on each frame (bool, default = true)
    :param round_to_power_of_two: If true, round window size to power of two by zero-padding input to FFT. (bool, default = true)
    :param sample_frequency: Waveform data sample frequency (must match the waveform file, if specified there) (float, default = 16000)
    :param snip_edges: If true, end effects will be handled by outputting only frames that completely fit in the file, and the number of frames depends on the frame-length.  If false, the number of frames depends only on the frame-shift, and we reflect the data at the ends. (bool, default = true)
    :param use_energy: Add an extra energy output. (bool, default = false)
    :param use_log_fbank: If true, produce log-filterbank, else produce linear. (bool, default = true)
    :param use_power: If true, use power, else use magnitude. (bool, default = true)
    :param window_type: Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"sine"|"blackmann") (string, default = "povey")
    :param dtype: Type of array (np.float32|np.float64) (dtype or string, default=np.float32)
    :return: (Log) Mel filter bank energies.
    """
    window_size = int(frame_length * sample_frequency * 0.001)
    window_shift = int(frame_shift * sample_frequency * 0.001)
    frames, log_energy = extract_window(
        waveform=waveform,
        blackman_coeff=blackman_coeff,
        dither=dither,
        window_size=window_size,
        window_shift=window_shift,
        preemphasis_coefficient=preemphasis_coefficient,
        raw_energy=raw_energy,
        remove_dc_offset=remove_dc_offset,
        snip_edges=snip_edges,
        window_type=window_type,
        dtype=dtype
    )
    if round_to_power_of_two:
        n = 1
        while n < window_size:
            n *= 2
    else:
        n = window_size
    if use_power:
        spectrum = compute_power_spectrum(frames, n)
    else:
        spectrum = compute_spectrum(frames, n)
    mel_banks = compute_mel_banks(
        num_bins=num_mel_bins,
        sample_frequency=sample_frequency,
        low_freq=low_freq,
        high_freq=high_freq,
        n=n
    ).astype(dtype)
    feat = np.dot(spectrum, mel_banks.T)
    if use_log_fbank:
        feat = np.log(feat.clip(min=np.finfo(dtype).eps))
    if use_energy:
        if energy_floor > 0.0:
            log_energy.clip(min=np.math.log(energy_floor))
        return feat, log_energy
    return feat

# ---------- compute-fbank-feats ----------


# ---------- compute-mfcc-feats ----------

def compute_mfcc_feats(
        waveform,
        blackman_coeff=0.42,
        cepstral_lifter=22,
        dither=1.0,
        energy_floor=0.0,
        frame_length=25,
        frame_shift=10,
        high_freq=0,
        low_freq=20,
        num_ceps=13,
        num_mel_bins=23,
        preemphasis_coefficient=0.97,
        raw_energy=True,
        remove_dc_offset=True,
        round_to_power_of_two=True,
        sample_frequency=16000,
        snip_edges=True,
        use_energy=True,
        window_type='povey',
        dtype=np.float32):
    """ Compute mel-frequency cepstral coefficients

    :param waveform: Input waveform.
    :param blackman_coeff: Constant coefficient for generalized Blackman window. (float, default = 0.42)
    :param cepstral_lifter: Constant that controls scaling of MFCCs (float, default = 22)
    :param dither: Dithering constant (0.0 means no dither). If you turn this off, you should set the --energy-floor option, e.g. to 1.0 or 0.1 (float, default = 1)
    :param energy_floor: Floor on energy (absolute, not relative) in MFCC computation. Only makes a difference if --use-energy=true; only necessary if --dither=0.0.  Suggested values: 0.1 or 1.0 (float, default = 0)
    :param frame_length: Frame length in milliseconds (float, default = 25)
    :param frame_shift: Frame shift in milliseconds (float, default = 10)
    :param high_freq: High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)
    :param low_freq: Low cutoff frequency for mel bins (float, default = 20)
    :param num_ceps: Number of cepstra in MFCC computation (including C0) (int, default = 13)
    :param num_mel_bins: Number of triangular mel-frequency bins (int, default = 23)
    :param preemphasis_coefficient: Coefficient for use in signal preemphasis (float, default = 0.97)
    :param raw_energy: If true, compute energy before preemphasis and windowing (bool, default = true)
    :param remove_dc_offset: Subtract mean from waveform on each frame (bool, default = true)
    :param round_to_power_of_two: If true, round window size to power of two by zero-padding input to FFT. (bool, default = true)
    :param sample_frequency: Waveform data sample frequency (must match the waveform file, if specified there) (float, default = 16000)
    :param snip_edges: If true, end effects will be handled by outputting only frames that completely fit in the file, and the number of frames depends on the frame-length.  If false, the number of frames depends only on the frame-shift, and we reflect the data at the ends. (bool, default = true)
    :param use_energy: Use energy (not C0) in MFCC computation (bool, default = true)
    :param window_type: Type of window ("hamming"|"hanning"|"povey"|"rectangular"|"sine"|"blackmann") (string, default = "povey")
    :param dtype: Type of array (np.float32|np.float64) (dtype or string, default=np.float32)
    :return: Mel-frequency cespstral coefficients.
    """
    feat, log_energy = compute_fbank_feats(
        waveform=waveform,
        blackman_coeff=blackman_coeff,
        dither=dither,
        energy_floor=energy_floor,
        frame_length=frame_length,
        frame_shift=frame_shift,
        high_freq=high_freq,
        low_freq=low_freq,
        num_mel_bins=num_mel_bins,
        preemphasis_coefficient=preemphasis_coefficient,
        raw_energy=raw_energy,
        remove_dc_offset=remove_dc_offset,
        round_to_power_of_two=round_to_power_of_two,
        sample_frequency=sample_frequency,
        snip_edges=snip_edges,
        use_energy=use_energy,
        use_log_fbank=True,
        use_power=True,
        window_type=window_type,
        dtype=dtype
    )
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :num_ceps]
    lifter_coeffs = compute_lifter_coeffs(cepstral_lifter, num_ceps).astype(dtype)
    feat = feat * lifter_coeffs
    if use_energy:
        feat[:, 0] = log_energy
    return feat

# ---------- compute-mfcc-feats ----------


# ---------- apply-cmvn-sliding ----------

def apply_cmvn_sliding(feat, center=False, window=600, min_window=100, norm_vars=False):
    """ Apply sliding-window cepstral mean (and optionally variance) normalization

    :param feat: Cepstrum.
    :param center: If true, use a window centered on the current frame (to the extent possible, modulo end effects). If false, window is to the left. (bool, default = false)
    :param window: Window in frames for running average CMN computation (int, default = 600)
    :param min_window: Minimum CMN window used at start of decoding (adds latency only at start). Only applicable if center == false, ignored if center==true (int, default = 100)
    :param norm_vars: If true, normalize variance to one. (bool, default = false)
    :return: Normalized cepstrum.
    """
    # double-precision
    feat = apply_cmvn_sliding_internal(
        feat=feat.astype(np.float64),
        center=center,
        window=window,
        min_window=min_window,
        norm_vars=norm_vars
    ).astype(feat.dtype)
    return feat

# ---------- apply-cmvn-sliding ----------
