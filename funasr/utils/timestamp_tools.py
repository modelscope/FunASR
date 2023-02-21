import torch
import copy
import logging
import numpy as np
from typing import Any, List, Tuple, Union

def time_stamp_lfr6_pl(us_alphas, us_cif_peak, char_list, begin_time=0.0, end_time=None):
    if not len(char_list):
        return []
    START_END_THRESHOLD = 5
    TIME_RATE = 10.0 * 6 / 1000 / 3  #  3 times upsampled
    if len(us_alphas.shape) == 3:
        alphas, cif_peak = us_alphas[0], us_cif_peak[0]  # support inference batch_size=1 only
    else:
        alphas, cif_peak = us_alphas, us_cif_peak
    num_frames = cif_peak.shape[0]
    if char_list[-1] == '</s>':
        char_list = char_list[:-1]
    # char_list = [i for i in text]
    timestamp_list = []
    # for bicif model trained with large data, cif2 actually fires when a character starts
    # so treat the frames between two peaks as the duration of the former token
    fire_place = torch.where(cif_peak>1.0-1e-4)[0].cpu().numpy() - 1.5
    num_peak = len(fire_place)
    assert num_peak == len(char_list) + 1 # number of peaks is supposed to be number of tokens + 1
    # begin silence
    if fire_place[0] > START_END_THRESHOLD:
        char_list.insert(0, '<sil>')
        timestamp_list.append([0.0, fire_place[0]*TIME_RATE])
    # tokens timestamp
    for i in range(len(fire_place)-1):
        # the peak is always a little ahead of the start time
        # timestamp_list.append([(fire_place[i]-1.2)*TIME_RATE, fire_place[i+1]*TIME_RATE])
        timestamp_list.append([(fire_place[i])*TIME_RATE, fire_place[i+1]*TIME_RATE])
        # cut the duration to token and sil of the 0-weight frames last long
    # tail token and end silence
    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) / 2
        timestamp_list[-1][1] = _end*TIME_RATE
        timestamp_list.append([_end*TIME_RATE, num_frames*TIME_RATE])
        char_list.append("<sil>")
    else:
        timestamp_list[-1][1] = num_frames*TIME_RATE
    if begin_time:  # add offset time in model with vad
        for i in range(len(timestamp_list)):
            timestamp_list[i][0] = timestamp_list[i][0] + begin_time / 1000.0
            timestamp_list[i][1] = timestamp_list[i][1] + begin_time / 1000.0
    res_txt = ""
    for char, timestamp in zip(char_list, timestamp_list):
        res_txt += "{} {} {};".format(char, timestamp[0], timestamp[1])
    res = []
    for char, timestamp in zip(char_list, timestamp_list):
        if char != '<sil>':
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res

