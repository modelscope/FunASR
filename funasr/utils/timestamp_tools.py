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


def time_stamp_lfr6_advance(tst: List, text: str):
    # advanced timestamp prediction for BiCIF_Paraformer using upsampled alphas
    ds_alphas, ds_cif_peak, us_alphas, us_cif_peak = tst
    if text.endswith('</s>'):
        text = text[:-4]
    else:
        text = text[:-1]
        logging.warning("found text does not end with </s>")
    assert int(ds_alphas.sum() + 1e-4) - 1 == len(text)
    
