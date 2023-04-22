#!/usr/bin/env python3
#
# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)


import pickle

import kaldi_native_fbank as knf


def test_default():
    opts = knf.FbankOptions()
    assert opts.frame_opts.samp_freq == 16000
    assert opts.frame_opts.frame_shift_ms == 10.0
    assert opts.frame_opts.frame_length_ms == 25.0
    assert opts.frame_opts.dither == 1.0
    assert abs(opts.frame_opts.preemph_coeff - 0.97) < 1e-6
    assert opts.frame_opts.remove_dc_offset is True
    assert opts.frame_opts.window_type == "povey"
    assert opts.frame_opts.round_to_power_of_two is True
    assert abs(opts.frame_opts.blackman_coeff - 0.42) < 1e-6
    assert opts.frame_opts.snip_edges is True

    assert opts.mel_opts.num_bins == 23
    assert opts.mel_opts.low_freq == 20
    assert opts.mel_opts.high_freq == 0
    assert opts.mel_opts.vtln_low == 100
    assert opts.mel_opts.vtln_high == -500
    assert opts.mel_opts.debug_mel is False
    assert opts.mel_opts.htk_mode is False

    assert opts.use_energy is False
    assert opts.energy_floor == 0.0
    assert opts.raw_energy is True
    assert opts.htk_compat is False
    assert opts.use_log_fbank is True
    assert opts.use_power is True


def test_set_get():
    opts = knf.FbankOptions()
    opts.use_energy = True
    assert opts.use_energy is True

    opts.energy_floor = 1
    assert opts.energy_floor == 1

    opts.raw_energy = False
    assert opts.raw_energy is False

    opts.htk_compat = True
    assert opts.htk_compat is True

    opts.use_log_fbank = False
    assert opts.use_log_fbank is False

    opts.use_power = False
    assert opts.use_power is False


def test_set_get_frame_opts():
    opts = knf.FbankOptions()

    opts.frame_opts.samp_freq = 44100
    assert opts.frame_opts.samp_freq == 44100

    opts.frame_opts.frame_shift_ms = 20.5
    assert opts.frame_opts.frame_shift_ms == 20.5

    opts.frame_opts.frame_length_ms = 1
    assert opts.frame_opts.frame_length_ms == 1

    opts.frame_opts.dither = 0.5
    assert opts.frame_opts.dither == 0.5

    opts.frame_opts.preemph_coeff = 0.25
    assert opts.frame_opts.preemph_coeff == 0.25

    opts.frame_opts.remove_dc_offset = False
    assert opts.frame_opts.remove_dc_offset is False

    opts.frame_opts.window_type = "hanning"
    assert opts.frame_opts.window_type == "hanning"

    opts.frame_opts.round_to_power_of_two = False
    assert opts.frame_opts.round_to_power_of_two is False

    opts.frame_opts.blackman_coeff = 0.25
    assert opts.frame_opts.blackman_coeff == 0.25

    opts.frame_opts.snip_edges = False
    assert opts.frame_opts.snip_edges is False


def test_set_get_mel_opts():
    opts = knf.FbankOptions()

    opts.mel_opts.num_bins = 100
    assert opts.mel_opts.num_bins == 100

    opts.mel_opts.low_freq = 22
    assert opts.mel_opts.low_freq == 22

    opts.mel_opts.high_freq = 1
    assert opts.mel_opts.high_freq == 1

    opts.mel_opts.vtln_low = 101
    assert opts.mel_opts.vtln_low == 101

    opts.mel_opts.vtln_high = -100
    assert opts.mel_opts.vtln_high == -100

    opts.mel_opts.debug_mel = True
    assert opts.mel_opts.debug_mel is True

    opts.mel_opts.htk_mode = True
    assert opts.mel_opts.htk_mode is True


def test_from_empty_dict():
    opts = knf.FbankOptions.from_dict({})
    opts2 = knf.FbankOptions()

    assert str(opts) == str(opts2)


def test_from_dict_partial():
    d = {
        "energy_floor": 10.5,
        "htk_compat": True,
        "mel_opts": {"num_bins": 80, "vtln_low": 1},
        "frame_opts": {"window_type": "hanning"},
    }
    opts = knf.FbankOptions.from_dict(d)
    assert opts.energy_floor == 10.5
    assert opts.htk_compat is True
    assert opts.mel_opts.num_bins == 80
    assert opts.mel_opts.vtln_low == 1
    assert opts.frame_opts.window_type == "hanning"

    mel_opts = knf.MelBanksOptions.from_dict(d["mel_opts"])
    assert str(opts.mel_opts) == str(mel_opts)


def test_from_dict_full_and_as_dict():
    opts = knf.FbankOptions()
    opts.htk_compat = True
    opts.mel_opts.num_bins = 80
    opts.frame_opts.samp_freq = 10

    d = opts.as_dict()
    assert d["htk_compat"] is True
    assert d["mel_opts"]["num_bins"] == 80
    assert d["frame_opts"]["samp_freq"] == 10

    mel_opts = knf.MelBanksOptions()
    mel_opts.num_bins = 80
    assert d["mel_opts"] == mel_opts.as_dict()

    frame_opts = knf.FrameExtractionOptions()
    frame_opts.samp_freq = 10
    assert d["frame_opts"] == frame_opts.as_dict()

    opts2 = knf.FbankOptions.from_dict(d)
    assert str(opts2) == str(opts)

    d["htk_compat"] = False
    opts3 = knf.FbankOptions.from_dict(d)
    assert opts3.htk_compat is False


def test_pickle():
    opts = knf.FbankOptions()
    opts.use_energy = True
    opts.use_power = False

    opts.frame_opts.samp_freq = 44100
    opts.mel_opts.num_bins = 100

    data = pickle.dumps(opts)

    opts2 = pickle.loads(data)
    assert str(opts) == str(opts2)


def main():
    test_default()
    test_set_get()
    test_set_get_frame_opts()
    test_set_get_mel_opts()
    test_from_empty_dict()
    test_from_dict_partial()
    test_from_dict_full_and_as_dict()
    test_pickle()


if __name__ == "__main__":
    main()
