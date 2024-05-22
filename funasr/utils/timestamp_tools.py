import torch
import codecs
import logging
import argparse
import numpy as np

# import edit_distance
from itertools import zip_longest


def cif_wo_hidden(alphas, threshold):
    batch_size, len_time = alphas.size()
    # loop varss
    integrate = torch.zeros([batch_size], device=alphas.device)
    # intermediate vars along time
    list_fires = []
    for t in range(len_time):
        alpha = alphas[:, t]
        integrate += alpha
        list_fires.append(integrate)
        fire_place = integrate >= threshold
        integrate = torch.where(
            fire_place,
            integrate - torch.ones([batch_size], device=alphas.device) * threshold,
            integrate,
        )
    fires = torch.stack(list_fires, 1)
    return fires


def ts_prediction_lfr6_standard(
    us_alphas, us_peaks, char_list, vad_offset=0.0, force_time_shift=-1.5, sil_in_str=True
):
    if not len(char_list):
        return "", []
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION = 12
    TIME_RATE = 10.0 * 6 / 1000 / 3  #  3 times upsampled
    if len(us_alphas.shape) == 2:
        alphas, peaks = us_alphas[0], us_peaks[0]  # support inference batch_size=1 only
    else:
        alphas, peaks = us_alphas, us_peaks
    if char_list[-1] == "</s>":
        char_list = char_list[:-1]
    fire_place = (
        torch.where(peaks >= 1.0 - 1e-4)[0].cpu().numpy() + force_time_shift
    )  # total offset
    if len(fire_place) != len(char_list) + 1:
        alphas /= alphas.sum() / (len(char_list) + 1)
        alphas = alphas.unsqueeze(0)
        peaks = cif_wo_hidden(alphas, threshold=1.0 - 1e-4)[0]
        fire_place = (
            torch.where(peaks >= 1.0 - 1e-4)[0].cpu().numpy() + force_time_shift
        )  # total offset
    num_frames = peaks.shape[0]
    timestamp_list = []
    new_char_list = []
    # for bicif model trained with large data, cif2 actually fires when a character starts
    # so treat the frames between two peaks as the duration of the former token
    # fire_place = torch.where(peaks>=1.0-1e-4)[0].cpu().numpy() + force_time_shift  # total offset
    # assert num_peak == len(char_list) + 1 # number of peaks is supposed to be number of tokens + 1
    # begin silence
    if fire_place[0] > START_END_THRESHOLD:
        # char_list.insert(0, '<sil>')
        timestamp_list.append([0.0, fire_place[0] * TIME_RATE])
        new_char_list.append("<sil>")
    # tokens timestamp
    for i in range(len(fire_place) - 1):
        new_char_list.append(char_list[i])
        if MAX_TOKEN_DURATION < 0 or fire_place[i + 1] - fire_place[i] <= MAX_TOKEN_DURATION:
            timestamp_list.append([fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE])
        else:
            # cut the duration to token and sil of the 0-weight frames last long
            _split = fire_place[i] + MAX_TOKEN_DURATION
            timestamp_list.append([fire_place[i] * TIME_RATE, _split * TIME_RATE])
            timestamp_list.append([_split * TIME_RATE, fire_place[i + 1] * TIME_RATE])
            new_char_list.append("<sil>")
    # tail token and end silence
    # new_char_list.append(char_list[-1])
    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) * 0.5
        # _end = fire_place[-1]
        timestamp_list[-1][1] = _end * TIME_RATE
        timestamp_list.append([_end * TIME_RATE, num_frames * TIME_RATE])
        new_char_list.append("<sil>")
    else:
        timestamp_list[-1][1] = num_frames * TIME_RATE
    if vad_offset:  # add offset time in model with vad
        for i in range(len(timestamp_list)):
            timestamp_list[i][0] = timestamp_list[i][0] + vad_offset / 1000.0
            timestamp_list[i][1] = timestamp_list[i][1] + vad_offset / 1000.0
    res_txt = ""
    for char, timestamp in zip(new_char_list, timestamp_list):
        # if char != '<sil>':
        if not sil_in_str and char == "<sil>":
            continue
        res_txt += "{} {} {};".format(
            char, str(timestamp[0] + 0.0005)[:5], str(timestamp[1] + 0.0005)[:5]
        )
    res = []
    for char, timestamp in zip(new_char_list, timestamp_list):
        if char != "<sil>":
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res_txt, res


def timestamp_sentence(
    punc_id_list, timestamp_postprocessed, text_postprocessed, return_raw_text=False
):
    punc_list = ["，", "。", "？", "、"]
    res = []
    if text_postprocessed is None:
        return res
    if timestamp_postprocessed is None:
        return res
    if len(timestamp_postprocessed) == 0:
        return res
    if len(text_postprocessed) == 0:
        return res

    if punc_id_list is None or len(punc_id_list) == 0:
        res.append(
            {
                "text": text_postprocessed.split(),
                "start": timestamp_postprocessed[0][0],
                "end": timestamp_postprocessed[-1][1],
                "timestamp": timestamp_postprocessed,
            }
        )
        return res
    if len(punc_id_list) != len(timestamp_postprocessed):
        logging.warning("length mismatch between punc and timestamp")
    sentence_text = ""
    sentence_text_seg = ""
    ts_list = []
    sentence_start = timestamp_postprocessed[0][0]
    sentence_end = timestamp_postprocessed[0][1]
    texts = text_postprocessed.split()
    punc_stamp_text_list = list(
        zip_longest(punc_id_list, timestamp_postprocessed, texts, fillvalue=None)
    )
    for punc_stamp_text in punc_stamp_text_list:
        punc_id, timestamp, text = punc_stamp_text
        # sentence_text += text if text is not None else ''
        if text is not None:
            if "a" <= text[0] <= "z" or "A" <= text[0] <= "Z":
                sentence_text += " " + text
            elif len(sentence_text) and (
                "a" <= sentence_text[-1] <= "z" or "A" <= sentence_text[-1] <= "Z"
            ):
                sentence_text += " " + text
            else:
                sentence_text += text
            sentence_text_seg += text + " "
        ts_list.append(timestamp)

        punc_id = int(punc_id) if punc_id is not None else 1
        sentence_end = timestamp[1] if timestamp is not None else sentence_end
        sentence_text_seg = (
            sentence_text_seg[:-1] if sentence_text_seg[-1] == " " else sentence_text_seg
        )
        if punc_id > 1:
            sentence_text += punc_list[punc_id - 2]
            if return_raw_text:
                res.append(
                    {
                        "text": sentence_text,
                        "start": sentence_start,
                        "end": sentence_end,
                        "timestamp": ts_list,
                        "raw_text": sentence_text_seg,
                    }
                )
            else:
                res.append(
                    {
                        "text": sentence_text,
                        "start": sentence_start,
                        "end": sentence_end,
                        "timestamp": ts_list,
                    }
                )
            sentence_text = ""
            sentence_text_seg = ""
            ts_list = []
            sentence_start = sentence_end
    return res
