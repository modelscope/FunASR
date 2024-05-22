# Copyright (c) Alibaba, Inc. and its affiliates.
""" Some implementations are adapted from https://github.com/yuyq96/D-TDNN
"""

import io
from typing import Union

import librosa as sf
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
from torch import nn

from funasr.utils.modelscope_file import File


def check_audio_list(audio: list):
    audio_dur = 0
    for i in range(len(audio)):
        seg = audio[i]
        assert seg[1] >= seg[0], "modelscope error: Wrong time stamps."
        assert isinstance(seg[2], np.ndarray), "modelscope error: Wrong data type."
        assert (
            int(seg[1] * 16000) - int(seg[0] * 16000) == seg[2].shape[0]
        ), "modelscope error: audio data in list is inconsistent with time length."
        if i > 0:
            assert seg[0] >= audio[i - 1][1], "modelscope error: Wrong time stamps."
        audio_dur += seg[1] - seg[0]
    return audio_dur
    # assert audio_dur > 5, 'modelscope error: The effective audio duration is too short.'


def sv_preprocess(inputs: Union[np.ndarray, list]):
    output = []
    for i in range(len(inputs)):
        if isinstance(inputs[i], str):
            file_bytes = File.read(inputs[i])
            data, fs = sf.load(io.BytesIO(file_bytes), dtype="float32")
            if len(data.shape) == 2:
                data = data[:, 0]
            data = torch.from_numpy(data).unsqueeze(0)
            data = data.squeeze(0)
        elif isinstance(inputs[i], np.ndarray):
            assert len(inputs[i].shape) == 1, "modelscope error: Input array should be [N, T]"
            data = inputs[i]
            if data.dtype in ["int16", "int32", "int64"]:
                data = (data / (1 << 15)).astype("float32")
            else:
                data = data.astype("float32")
            data = torch.from_numpy(data)
        else:
            raise ValueError(
                "modelscope error: The input type is restricted to audio address and nump array."
            )
        output.append(data)
    return output


def sv_chunk(vad_segments: list, fs=16000) -> list:
    config = {
        "seg_dur": 1.5,
        "seg_shift": 0.75,
    }

    def seg_chunk(seg_data):
        seg_st = seg_data[0]
        data = seg_data[2]
        chunk_len = int(config["seg_dur"] * fs)
        chunk_shift = int(config["seg_shift"] * fs)
        last_chunk_ed = 0
        seg_res = []
        for chunk_st in range(0, data.shape[0], chunk_shift):
            chunk_ed = min(chunk_st + chunk_len, data.shape[0])
            if chunk_ed <= last_chunk_ed:
                break
            last_chunk_ed = chunk_ed
            chunk_st = max(0, chunk_ed - chunk_len)
            chunk_data = data[chunk_st:chunk_ed]
            if chunk_data.shape[0] < chunk_len:
                chunk_data = np.pad(chunk_data, (0, chunk_len - chunk_data.shape[0]), "constant")
            seg_res.append([chunk_st / fs + seg_st, chunk_ed / fs + seg_st, chunk_data])
        return seg_res

    segs = []
    for i, s in enumerate(vad_segments):
        segs.extend(seg_chunk(s))

    return segs


def extract_feature(audio):
    features = []
    for au in audio:
        feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature.unsqueeze(0))
    features = torch.cat(features)
    return features


def postprocess(
    segments: list, vad_segments: list, labels: np.ndarray, embeddings: np.ndarray
) -> list:
    assert len(segments) == len(labels)
    labels = correct_labels(labels)
    distribute_res = []
    for i in range(len(segments)):
        distribute_res.append([segments[i][0], segments[i][1], labels[i]])
    # merge the same speakers chronologically
    distribute_res = merge_seque(distribute_res)

    # accquire speaker center
    spk_embs = []
    for i in range(labels.max() + 1):
        spk_emb = embeddings[labels == i].mean(0)
        spk_embs.append(spk_emb)
    spk_embs = np.stack(spk_embs)

    def is_overlapped(t1, t2):
        if t1 > t2 + 1e-4:
            return True
        return False

    # distribute the overlap region
    for i in range(1, len(distribute_res)):
        if is_overlapped(distribute_res[i - 1][1], distribute_res[i][0]):
            p = (distribute_res[i][0] + distribute_res[i - 1][1]) / 2
            distribute_res[i][0] = p
            distribute_res[i - 1][1] = p

    # smooth the result
    distribute_res = smooth(distribute_res)

    return distribute_res


def correct_labels(labels):
    labels_id = 0
    id2id = {}
    new_labels = []
    for i in labels:
        if i not in id2id:
            id2id[i] = labels_id
            labels_id += 1
        new_labels.append(id2id[i])
    return np.array(new_labels)


def merge_seque(distribute_res):
    res = [distribute_res[0]]
    for i in range(1, len(distribute_res)):
        if distribute_res[i][2] != res[-1][2] or distribute_res[i][0] > res[-1][1]:
            res.append(distribute_res[i])
        else:
            res[-1][1] = distribute_res[i][1]
    return res


def smooth(res, mindur=1):
    # short segments are assigned to nearest speakers.
    for i in range(len(res)):
        res[i][0] = round(res[i][0], 2)
        res[i][1] = round(res[i][1], 2)
        if res[i][1] - res[i][0] < mindur:
            if i == 0:
                res[i][2] = res[i + 1][2]
            elif i == len(res) - 1:
                res[i][2] = res[i - 1][2]
            elif res[i][0] - res[i - 1][1] <= res[i + 1][0] - res[i][1]:
                res[i][2] = res[i - 1][2]
            else:
                res[i][2] = res[i + 1][2]
    # merge the speakers
    res = merge_seque(res)

    return res


def distribute_spk(sentence_list, sd_time_list):
    sd_sentence_list = []
    for d in sentence_list:
        sentence_start = d["ts_list"][0][0]
        sentence_end = d["ts_list"][-1][1]
        sentence_spk = 0
        max_overlap = 0
        for sd_time in sd_time_list:
            spk_st, spk_ed, spk = sd_time
            spk_st = spk_st * 1000
            spk_ed = spk_ed * 1000
            overlap = max(min(sentence_end, spk_ed) - max(sentence_start, spk_st), 0)
            if overlap > max_overlap:
                max_overlap = overlap
                sentence_spk = spk
        d["spk"] = sentence_spk
        sd_sentence_list.append(d)
    return sd_sentence_list
