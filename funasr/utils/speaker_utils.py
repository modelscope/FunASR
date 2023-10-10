# Copyright (c) Alibaba, Inc. and its affiliates.
""" Some implementations are adapted from https://github.com/yuyq96/D-TDNN
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn

import io
import os
from typing import Any, Dict, List, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
import logging
from funasr.utils.modelscope_file import File
from collections import OrderedDict
import torchaudio.compliance.kaldi as Kaldi


def check_audio_list(audio: list):
    audio_dur = 0
    for i in range(len(audio)):
        seg = audio[i]
        assert seg[1] >= seg[0], 'modelscope error: Wrong time stamps.'
        assert isinstance(seg[2], np.ndarray), 'modelscope error: Wrong data type.'
        assert int(seg[1] * 16000) - int(
            seg[0] * 16000
        ) == seg[2].shape[
            0], 'modelscope error: audio data in list is inconsistent with time length.'
        if i > 0:
            assert seg[0] >= audio[
                i - 1][1], 'modelscope error: Wrong time stamps.'
        audio_dur += seg[1] - seg[0]
    assert audio_dur > 5, 'modelscope error: The effective audio duration is too short.'


def sv_preprocess(inputs: Union[np.ndarray, list]):
        output = []
        for i in range(len(inputs)):
            if isinstance(inputs[i], str):
                file_bytes = File.read(inputs[i])
                data, fs = sf.read(io.BytesIO(file_bytes), dtype='float32')
                if len(data.shape) == 2:
                    data = data[:, 0]
                data = torch.from_numpy(data).unsqueeze(0)
                data = data.squeeze(0)
            elif isinstance(inputs[i], np.ndarray):
                assert len(
                    inputs[i].shape
                ) == 1, 'modelscope error: Input array should be [N, T]'
                data = inputs[i]
                if data.dtype in ['int16', 'int32', 'int64']:
                    data = (data / (1 << 15)).astype('float32')
                else:
                    data = data.astype('float32')
                data = torch.from_numpy(data)
            else:
                raise ValueError(
                    'modelscope error: The input type is restricted to audio address and nump array.'
                )
            output.append(data)
        return output


def sv_chunk(vad_segments: list, fs = 16000) -> list:
    config = {
            'seg_dur': 1.5,
            'seg_shift': 0.75,
        }
    def seg_chunk(seg_data):
        seg_st = seg_data[0]
        data = seg_data[2]
        chunk_len = int(config['seg_dur'] * fs)
        chunk_shift = int(config['seg_shift'] * fs)
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
                chunk_data = np.pad(chunk_data,
                                    (0, chunk_len - chunk_data.shape[0]),
                                    'constant')
            seg_res.append([
                chunk_st / fs + seg_st, chunk_ed / fs + seg_st,
                chunk_data
            ])
        return seg_res

    segs = []
    for i, s in enumerate(vad_segments):
        segs.extend(seg_chunk(s))

    return segs


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=(stride, 1),
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(nn.Module):

    def __init__(self,
                 block=BasicResBlock,
                 num_blocks=[2, 2],
                 m_channels=32,
                 feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(
            block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(
            block, m_channels, num_blocks[0], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels,
            m_channels,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):

    def __init__(self,
                 feat_dim=80,
                 embedding_size=192,
                 growth_rate=32,
                 bn_size=4,
                 init_channels=128,
                 config_str='batchnorm-relu',
                 memory_efficient=True,
                 output_level='segment'):
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = nn.Sequential(
            OrderedDict([
                ('tdnn',
                 TDNNLayer(
                     channels,
                     init_channels,
                     5,
                     stride=2,
                     dilation=1,
                     padding=-1,
                     config_str=config_str)),
            ]))
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
                zip((12, 24, 16), (3, 3, 3), (1, 2, 2))):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient)
            self.xvector.add_module('block%d' % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                'transit%d' % (i + 1),
                TransitLayer(
                    channels, channels // 2, bias=False,
                    config_str=config_str))
            channels //= 2

        self.xvector.add_module('out_nonlinear',
                                get_nonlinear(config_str, channels))

        if self.output_level == 'segment':
            self.xvector.add_module('stats', StatsPool())
            self.xvector.add_module(
                'dense',
                DenseLayer(
                    channels * 2, embedding_size, config_str='batchnorm_'))
        else:
            assert self.output_level == 'frame', '`output_level` should be set to \'segment\' or \'frame\'. '

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        if self.output_level == 'frame':
            x = x.transpose(1, 2)
        return x


def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):

    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
                kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


def extract_feature(audio):
    features = []
    for au in audio:
        feature = Kaldi.fbank(
            au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature.unsqueeze(0))
    features = torch.cat(features)
    return features


class CAMLayer(nn.Module):

    def __init__(self,
                 bn_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias,
                 reduction=2):
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype='avg'):
        if stype == 'avg':
            seg = F.avg_pool1d(
                x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == 'max':
            seg = F.max_pool1d(
                x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError('Wrong segment pooling type.')
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape,
                                       seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):

    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 bn_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 config_str='batchnorm-relu',
                 memory_efficient=False):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient)
            self.add_module('tdnnd%d' % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 config_str='batchnorm-relu'):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 config_str='batchnorm-relu'):
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x

def postprocess(segments: list, vad_segments: list,
                labels: np.ndarray, embeddings: np.ndarray) -> list:
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
        if distribute_res[i][2] != res[-1][2] or distribute_res[i][
                0] > res[-1][1]:
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
        sentence_start = d['ts_list'][0][0]
        sentence_end = d['ts_list'][-1][1]
        sentence_spk = 0
        max_overlap = 0
        for sd_time in sd_time_list:
            spk_st, spk_ed, spk = sd_time
            spk_st = spk_st*1000
            spk_ed = spk_ed*1000
            overlap = max(
                min(sentence_end, spk_ed) - max(sentence_start, spk_st), 0)
            if overlap > max_overlap:
                max_overlap = overlap
                sentence_spk = spk
        d['spk'] = sentence_spk
        sd_sentence_list.append(d)
    return sd_sentence_list