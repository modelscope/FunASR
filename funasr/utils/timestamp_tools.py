import torch
import copy
import logging
import numpy as np
from typing import Any, List, Tuple, Union


def ts_prediction_lfr6_standard(us_alphas, 
                       us_peaks, 
                       char_list, 
                       vad_offset=0.0, 
                       force_time_shift=-1.5
                       ):
    if not len(char_list):
        return []
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION = 12
    TIME_RATE = 10.0 * 6 / 1000 / 3  #  3 times upsampled
    if len(us_alphas.shape) == 2:
        _, peaks = us_alphas[0], us_peaks[0]  # support inference batch_size=1 only
    else:
        _, peaks = us_alphas, us_peaks
    num_frames = peaks.shape[0]
    if char_list[-1] == '</s>':
        char_list = char_list[:-1]
    timestamp_list = []
    new_char_list = []
    # for bicif model trained with large data, cif2 actually fires when a character starts
    # so treat the frames between two peaks as the duration of the former token
    fire_place = torch.where(peaks>1.0-1e-4)[0].cpu().numpy() + force_time_shift  # total offset
    num_peak = len(fire_place)
    assert num_peak == len(char_list) + 1 # number of peaks is supposed to be number of tokens + 1
    # begin silence
    if fire_place[0] > START_END_THRESHOLD:
        # char_list.insert(0, '<sil>')
        timestamp_list.append([0.0, fire_place[0]*TIME_RATE])
        new_char_list.append('<sil>')
    # tokens timestamp
    for i in range(len(fire_place)-1):
        new_char_list.append(char_list[i])
        if MAX_TOKEN_DURATION < 0 or fire_place[i+1] - fire_place[i] <= MAX_TOKEN_DURATION:
            timestamp_list.append([fire_place[i]*TIME_RATE, fire_place[i+1]*TIME_RATE])
        else:
            # cut the duration to token and sil of the 0-weight frames last long
            _split = fire_place[i] + MAX_TOKEN_DURATION
            timestamp_list.append([fire_place[i]*TIME_RATE, _split*TIME_RATE])
            timestamp_list.append([_split*TIME_RATE, fire_place[i+1]*TIME_RATE])
            new_char_list.append('<sil>')
    # tail token and end silence
    # new_char_list.append(char_list[-1])
    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) * 0.5
        # _end = fire_place[-1] 
        timestamp_list[-1][1] = _end*TIME_RATE
        timestamp_list.append([_end*TIME_RATE, num_frames*TIME_RATE])
        new_char_list.append("<sil>")
    else:
        timestamp_list[-1][1] = num_frames*TIME_RATE
    if vad_offset:  # add offset time in model with vad
        for i in range(len(timestamp_list)):
            timestamp_list[i][0] = timestamp_list[i][0] + vad_offset / 1000.0
            timestamp_list[i][1] = timestamp_list[i][1] + vad_offset / 1000.0
    res_txt = ""
    for char, timestamp in zip(new_char_list, timestamp_list):
        res_txt += "{} {} {};".format(char, str(timestamp[0]+0.0005)[:5], str(timestamp[1]+0.0005)[:5])
    res = []
    for char, timestamp in zip(new_char_list, timestamp_list):
        if char != '<sil>':
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res_txt, res


def time_stamp_sentence(punc_id_list, time_stamp_postprocessed, text_postprocessed):
    res = []
    if text_postprocessed is None:
        return res
    if time_stamp_postprocessed is None:
        return res
    if len(time_stamp_postprocessed) == 0:
        return res
    if len(text_postprocessed) == 0:
        return res
    if punc_id_list is None or len(punc_id_list) == 0:
        res.append({
            'text': text_postprocessed.split(),
            "start": time_stamp_postprocessed[0][0],
            "end": time_stamp_postprocessed[-1][1]
        })
        return res
    if len(punc_id_list) != len(time_stamp_postprocessed):
        res.append({
            'text': text_postprocessed.split(),
            "start": time_stamp_postprocessed[0][0],
            "end": time_stamp_postprocessed[-1][1]
        })
        return res

    sentence_text = ''
    sentence_start = time_stamp_postprocessed[0][0]
    texts = text_postprocessed.split()
    for i in range(len(punc_id_list)):
        sentence_text += texts[i]
        if punc_id_list[i] == 2:
            sentence_text += ','
            res.append({
                'text': sentence_text,
                "start": sentence_start,
                "end": time_stamp_postprocessed[i][1]
            })
            sentence_text = ''
            sentence_start = time_stamp_postprocessed[i][1]
        elif punc_id_list[i] == 3:
            sentence_text += '.'
            res.append({
                'text': sentence_text,
                "start": sentence_start,
                "end": time_stamp_postprocessed[i][1]
            })
            sentence_text = ''
            sentence_start = time_stamp_postprocessed[i][1]
    return res



