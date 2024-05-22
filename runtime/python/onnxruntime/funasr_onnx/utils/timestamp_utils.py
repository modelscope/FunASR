# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import numpy as np


def time_stamp_lfr6_onnx(us_cif_peak, char_list, begin_time=0.0, total_offset=-1.5):
    if not len(char_list):
        return []
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION = 30
    TIME_RATE = 10.0 * 6 / 1000 / 3  #  3 times upsampled
    cif_peak = us_cif_peak.reshape(-1)
    num_frames = cif_peak.shape[-1]
    if char_list[-1] == "</s>":
        char_list = char_list[:-1]
    # char_list = [i for i in text]
    timestamp_list = []
    new_char_list = []
    # for bicif model trained with large data, cif2 actually fires when a character starts
    # so treat the frames between two peaks as the duration of the former token
    fire_place = np.where(cif_peak > 1.0 - 1e-4)[0] + total_offset  # np format
    num_peak = len(fire_place)
    assert num_peak == len(char_list) + 1  # number of peaks is supposed to be number of tokens + 1
    # begin silence
    if fire_place[0] > START_END_THRESHOLD:
        # char_list.insert(0, '<sil>')
        timestamp_list.append([0.0, fire_place[0] * TIME_RATE])
        new_char_list.append("<sil>")
    # tokens timestamp
    for i in range(len(fire_place) - 1):
        new_char_list.append(char_list[i])
        if (
            i == len(fire_place) - 2
            or MAX_TOKEN_DURATION < 0
            or fire_place[i + 1] - fire_place[i] < MAX_TOKEN_DURATION
        ):
            timestamp_list.append([fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE])
        else:
            # cut the duration to token and sil of the 0-weight frames last long
            _split = fire_place[i] + MAX_TOKEN_DURATION
            timestamp_list.append([fire_place[i] * TIME_RATE, _split * TIME_RATE])
            timestamp_list.append([_split * TIME_RATE, fire_place[i + 1] * TIME_RATE])
            new_char_list.append("<sil>")
    # tail token and end silence
    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) / 2
        timestamp_list[-1][1] = _end * TIME_RATE
        timestamp_list.append([_end * TIME_RATE, num_frames * TIME_RATE])
        new_char_list.append("<sil>")
    else:
        timestamp_list[-1][1] = num_frames * TIME_RATE
    if begin_time:  # add offset time in model with vad
        for i in range(len(timestamp_list)):
            timestamp_list[i][0] = timestamp_list[i][0] + begin_time / 1000.0
            timestamp_list[i][1] = timestamp_list[i][1] + begin_time / 1000.0
    assert len(new_char_list) == len(timestamp_list)
    res_str = ""
    for char, timestamp in zip(new_char_list, timestamp_list):
        res_str += "{} {} {};".format(char, timestamp[0], timestamp[1])
    res = []
    for char, timestamp in zip(new_char_list, timestamp_list):
        if char != "<sil>":
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res_str, res
