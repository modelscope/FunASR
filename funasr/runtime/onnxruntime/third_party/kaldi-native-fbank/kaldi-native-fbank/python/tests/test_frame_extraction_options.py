#!/usr/bin/env python3
#
# Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)

import pickle

import kaldi_native_fbank as knf


def test_default():
    opts = knf.FrameExtractionOptions()
    assert opts.samp_freq == 16000
    assert opts.frame_shift_ms == 10.0
    assert opts.frame_length_ms == 25.0
    assert opts.dither == 1.0
    assert abs(opts.preemph_coeff - 0.97) < 1e-6
    assert opts.remove_dc_offset is True
    assert opts.window_type == "povey"
    assert opts.round_to_power_of_two is True
    assert abs(opts.blackman_coeff - 0.42) < 1e-6
    assert opts.snip_edges is True


def test_set_get():
    opts = knf.FrameExtractionOptions()
    opts.samp_freq = 44100
    assert opts.samp_freq == 44100

    opts.frame_shift_ms = 20.5
    assert opts.frame_shift_ms == 20.5

    opts.frame_length_ms = 1
    assert opts.frame_length_ms == 1

    opts.dither = 0.5
    assert opts.dither == 0.5

    opts.preemph_coeff = 0.25
    assert opts.preemph_coeff == 0.25

    opts.remove_dc_offset = False
    assert opts.remove_dc_offset is False

    opts.window_type = "hanning"
    assert opts.window_type == "hanning"

    opts.round_to_power_of_two = False
    assert opts.round_to_power_of_two is False

    opts.blackman_coeff = 0.25
    assert opts.blackman_coeff == 0.25

    opts.snip_edges = False
    assert opts.snip_edges is False


def test_from_empty_dict():
    opts = knf.FrameExtractionOptions.from_dict({})
    opts2 = knf.FrameExtractionOptions()

    assert str(opts) == str(opts2)


def test_from_dict_partial():
    d = {"samp_freq": 10, "frame_shift_ms": 2}

    opts = knf.FrameExtractionOptions.from_dict(d)

    opts2 = knf.FrameExtractionOptions()
    assert str(opts) != str(opts2)

    opts2.samp_freq = 10
    assert str(opts) != str(opts2)

    opts2.frame_shift_ms = 2
    assert str(opts) == str(opts2)

    opts2.frame_shift_ms = 3
    assert str(opts) != str(opts2)


def test_from_dict_full_and_as_dict():
    opts = knf.FrameExtractionOptions()
    opts.samp_freq = 20
    opts.frame_length_ms = 100

    d = opts.as_dict()
    for key, value in d.items():
        assert value == getattr(opts, key)

    opts2 = knf.FrameExtractionOptions.from_dict(d)
    assert str(opts2) == str(opts)

    d["window_type"] = "hanning"
    opts3 = knf.FrameExtractionOptions.from_dict(d)
    assert opts3.window_type == "hanning"


def test_pickle():
    opts = knf.FrameExtractionOptions()
    opts.samp_freq = 44100
    opts.dither = 5.5
    data = pickle.dumps(opts)

    opts2 = pickle.loads(data)
    assert str(opts) == str(opts2)


def main():
    test_default()
    test_set_get()
    test_from_empty_dict()
    test_from_dict_partial()
    test_from_dict_full_and_as_dict()
    test_pickle()


if __name__ == "__main__":
    main()
