import torch
import copy
import logging
import numpy as np
from typing import Any, List, Tuple, Union

def cut_interval(alphas: torch.Tensor, start: int, end: int, tail: bool):
    if not tail:
        if end == start + 1:
            cut = (end + start) / 2.0
        else:
            alpha = alphas[start+1: end].tolist()
            reverse_steps = 1
            for reverse_alpha in alpha[::-1]:
                if reverse_alpha > 0.35:
                    reverse_steps += 1
                else:
                    break
            cut = end - reverse_steps
    else:
        if end != len(alphas) - 1:
            cut = end + 1
        else:
            cut = start + 1
    return float(cut)

def time_stamp_lfr6(alphas: torch.Tensor, speech_lengths: torch.Tensor, raw_text: List[str], begin: int = 0, end: int = None):
    time_stamp_list = []
    alphas = alphas[0]
    text = copy.deepcopy(raw_text)
    if end is None:
        time = speech_lengths * 60 / 1000
        sacle_rate = (time / speech_lengths[0]).tolist()
    else:
        time = (end - begin) / 1000
        sacle_rate = (time / speech_lengths[0]).tolist()

    predictor = (alphas > 0.5).int()
    fire_places = torch.nonzero(predictor == 1).squeeze(1).tolist()
    
    cuts = []
    npeak = int(predictor.sum())
    nchar = len(raw_text)
    if npeak - 1 == nchar:
        fire_places = torch.where((alphas > 0.5) == 1)[0].tolist()
        for i in range(len(fire_places)):
            if fire_places[i] < len(alphas) - 1:
                if 0.05 < alphas[fire_places[i]+1] < 0.5:
                    fire_places[i] += 1
    elif npeak < nchar:
        lost_num = nchar - npeak
        lost_fire = speech_lengths[0].tolist() - fire_places[-1]
        interval_distance = lost_fire // (lost_num + 1)
        for i in range(1, lost_num + 1):
            fire_places.append(fire_places[-1] + interval_distance)
    elif npeak - 1 > nchar:
        redundance_num = npeak - 1 - nchar
        for i in range(redundance_num):
            fire_places.pop() 

    cuts.append(0)
    start_sil = True
    if start_sil:
        text.insert(0, '<sil>')

    for i in range(len(fire_places)-1):
        cuts.append(cut_interval(alphas, fire_places[i], fire_places[i+1], tail=(i==len(fire_places)-2)))

    for i in range(2, len(fire_places)-2):
        if fire_places[i-2] == fire_places[i-1] - 1 and fire_places[i-1] != fire_places[i] - 1:
            cuts[i-1] += 1

    if cuts[-1] != len(alphas) - 1:
        text.append('<sil>')
        cuts.append(speech_lengths[0].tolist())
    cuts.insert(-1, (cuts[-1] + cuts[-2]) * 0.5)
    sec_fire_places = np.array(cuts) * sacle_rate
    for i in range(1, len(sec_fire_places) - 1):
        start, end = sec_fire_places[i], sec_fire_places[i+1]
        if i == len(sec_fire_places) - 2:
            end = time
        time_stamp_list.append([int(round(start, 2) * 1000) + begin, int(round(end, 2) * 1000) + begin])
        text = text[1:]
    if npeak - 1 == nchar or npeak > nchar:
        return time_stamp_list[:-1]
    else:
        return time_stamp_list

def time_stamp_lfr6_pl(us_alphas, us_cif_peak, char_list, begin_time=0.0, end_time=None):
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
    logging.warning(res_txt)  # for test
    res = []
    for char, timestamp in zip(char_list, timestamp_list):
        if char != '<sil>':
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res

