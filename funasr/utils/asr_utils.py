# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import struct
from typing import Any, Dict, List, Union

import torchaudio
import numpy as np
import pkg_resources
from modelscope.utils.logger import get_logger

logger = get_logger()

green_color = '\033[1;32m'
red_color = '\033[0;31;40m'
yellow_color = '\033[0;33;40m'
end_color = '\033[0m'

global_asr_language = 'zh-cn'

SUPPORT_AUDIO_TYPE_SETS = ['flac', 'mp3', 'ogg', 'opus', 'wav', 'pcm']

def get_version():
    return float(pkg_resources.get_distribution('easyasr').version)


def sample_rate_checking(audio_in: Union[str, bytes], audio_format: str):
    r_audio_fs = None

    if audio_format == 'wav' or audio_format == 'scp':
        r_audio_fs = get_sr_from_wav(audio_in)
    elif audio_format == 'pcm' and isinstance(audio_in, bytes):
        r_audio_fs = get_sr_from_bytes(audio_in)

    return r_audio_fs


def type_checking(audio_in: Union[str, bytes],
                  audio_fs: int = None,
                  recog_type: str = None,
                  audio_format: str = None):
    r_recog_type = recog_type
    r_audio_format = audio_format
    r_wav_path = audio_in

    if isinstance(audio_in, str):
        assert os.path.exists(audio_in), f'wav_path:{audio_in} does not exist'
    elif isinstance(audio_in, bytes):
        assert len(audio_in) > 0, 'audio in is empty'
        r_audio_format = 'pcm'
        r_recog_type = 'wav'

    if audio_in is None:
        # for raw_inputs
        r_recog_type = 'wav'
        r_audio_format = 'pcm'

    if r_recog_type is None and audio_in is not None:
        # audio_in is wav, recog_type is wav_file
        if os.path.isfile(audio_in):
            audio_type = os.path.basename(audio_in).lower()
            for support_audio_type in SUPPORT_AUDIO_TYPE_SETS:
                if audio_type.rfind(".{}".format(support_audio_type)) >= 0:
                    r_recog_type = 'wav'
                    r_audio_format = 'wav'
            if audio_type.rfind(".scp") >= 0:
                r_recog_type = 'wav'
                r_audio_format = 'scp'
            if r_recog_type is None:
                raise NotImplementedError(
                    f'Not supported audio type: {audio_type}')

        # recog_type is datasets_file
        elif os.path.isdir(audio_in):
            dir_name = os.path.basename(audio_in)
            if 'test' in dir_name:
                r_recog_type = 'test'
            elif 'dev' in dir_name:
                r_recog_type = 'dev'
            elif 'train' in dir_name:
                r_recog_type = 'train'

    if r_audio_format is None:
        if find_file_by_ends(audio_in, '.ark'):
            r_audio_format = 'kaldi_ark'
        elif find_file_by_ends(audio_in, '.wav') or find_file_by_ends(
                audio_in, '.WAV'):
            r_audio_format = 'wav'
        elif find_file_by_ends(audio_in, '.records'):
            r_audio_format = 'tfrecord'

    if r_audio_format == 'kaldi_ark' and r_recog_type != 'wav':
        # datasets with kaldi_ark file
        r_wav_path = os.path.abspath(os.path.join(r_wav_path, '../'))
    elif r_audio_format == 'tfrecord' and r_recog_type != 'wav':
        # datasets with tensorflow records file
        r_wav_path = os.path.abspath(os.path.join(r_wav_path, '../'))
    elif r_audio_format == 'wav' and r_recog_type != 'wav':
        # datasets with waveform files
        r_wav_path = os.path.abspath(os.path.join(r_wav_path, '../../'))

    return r_recog_type, r_audio_format, r_wav_path


def get_sr_from_bytes(wav: bytes):
    sr = None
    data = wav
    if len(data) > 44:
        try:
            header_fields = {}
            header_fields['ChunkID'] = str(data[0:4], 'UTF-8')
            header_fields['Format'] = str(data[8:12], 'UTF-8')
            header_fields['Subchunk1ID'] = str(data[12:16], 'UTF-8')
            if header_fields['ChunkID'] == 'RIFF' and header_fields[
                    'Format'] == 'WAVE' and header_fields[
                        'Subchunk1ID'] == 'fmt ':
                header_fields['SampleRate'] = struct.unpack('<I',
                                                            data[24:28])[0]
                sr = header_fields['SampleRate']
        except Exception:
            # no treatment
            pass
    else:
        logger.warn('audio bytes is ' + str(len(data)) + ' is invalid.')

    return sr


def get_sr_from_wav(fname: str):
    fs = None
    if os.path.isfile(fname):
        audio_type = os.path.basename(fname).lower()
        for support_audio_type in SUPPORT_AUDIO_TYPE_SETS:
            if audio_type.rfind(".{}".format(support_audio_type)) >= 0:
                if support_audio_type == "pcm":
                    fs = None
                else:
                    audio, fs = torchaudio.load(fname)
                break
        if audio_type.rfind(".scp") >= 0:
            with open(fname, encoding="utf-8") as f:
                for line in f:
                    wav_path = line.split()[1]
                    fs = get_sr_from_wav(wav_path)
                    if fs is not None:
                        break
        return fs
    elif os.path.isdir(fname):
        dir_files = os.listdir(fname)
        for file in dir_files:
            file_path = os.path.join(fname, file)
            if os.path.isfile(file_path):
                fs = get_sr_from_wav(file_path)
            elif os.path.isdir(file_path):
                fs = get_sr_from_wav(file_path)

            if fs is not None:
                break

    return fs


def find_file_by_ends(dir_path: str, ends: str):
    dir_files = os.listdir(dir_path)
    for file in dir_files:
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            if ends == ".wav" or ends == ".WAV":
                audio_type = os.path.basename(file_path).lower()
                for support_audio_type in SUPPORT_AUDIO_TYPE_SETS:
                    if audio_type.rfind(".{}".format(support_audio_type)) >= 0:
                        return True
                raise NotImplementedError(
                    f'Not supported audio type: {audio_type}')
            elif file_path.endswith(ends):
                return True
        elif os.path.isdir(file_path):
            if find_file_by_ends(file_path, ends):
                return True

    return False


def recursion_dir_all_wav(wav_list, dir_path: str) -> List[str]:
    dir_files = os.listdir(dir_path)
    for file in dir_files:
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            audio_type = os.path.basename(file_path).lower()
            for support_audio_type in SUPPORT_AUDIO_TYPE_SETS:
                if audio_type.rfind(".{}".format(support_audio_type)) >= 0:
                    wav_list.append(file_path)
        elif os.path.isdir(file_path):
            recursion_dir_all_wav(wav_list, file_path)

    return wav_list

def compute_wer(hyp_list: List[Any],
                ref_list: List[Any],
                lang: str = None) -> Dict[str, Any]:
    assert len(hyp_list) > 0, 'hyp list is empty'
    assert len(ref_list) > 0, 'ref list is empty'

    rst = {
        'Wrd': 0,
        'Corr': 0,
        'Ins': 0,
        'Del': 0,
        'Sub': 0,
        'Snt': 0,
        'Err': 0.0,
        'S.Err': 0.0,
        'wrong_words': 0,
        'wrong_sentences': 0
    }

    if lang is None:
        lang = global_asr_language

    for h_item in hyp_list:
        for r_item in ref_list:
            if h_item['key'] == r_item['key']:
                out_item = compute_wer_by_line(h_item['value'],
                                               r_item['value'],
                                               lang)
                rst['Wrd'] += out_item['nwords']
                rst['Corr'] += out_item['cor']
                rst['wrong_words'] += out_item['wrong']
                rst['Ins'] += out_item['ins']
                rst['Del'] += out_item['del']
                rst['Sub'] += out_item['sub']
                rst['Snt'] += 1
                if out_item['wrong'] > 0:
                    rst['wrong_sentences'] += 1
                    print_wrong_sentence(key=h_item['key'],
                                         hyp=h_item['value'],
                                         ref=r_item['value'])
                else:
                    print_correct_sentence(key=h_item['key'],
                                           hyp=h_item['value'],
                                           ref=r_item['value'])

                break

    if rst['Wrd'] > 0:
        rst['Err'] = round(rst['wrong_words'] * 100 / rst['Wrd'], 2)
    if rst['Snt'] > 0:
        rst['S.Err'] = round(rst['wrong_sentences'] * 100 / rst['Snt'], 2)

    return rst


def compute_wer_by_line(hyp: List[str],
                        ref: List[str],
                        lang: str = 'zh-cn') -> Dict[str, Any]:
    if lang != 'zh-cn':
        hyp = hyp.split()
        ref = ref.split()

    hyp = list(map(lambda x: x.lower(), hyp))
    ref = list(map(lambda x: x.lower(), ref))

    len_hyp = len(hyp)
    len_ref = len(ref)

    cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

    ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

    for i in range(len_hyp + 1):
        cost_matrix[i][0] = i
    for j in range(len_ref + 1):
        cost_matrix[0][j] = j

    for i in range(1, len_hyp + 1):
        for j in range(1, len_ref + 1):
            if hyp[i - 1] == ref[j - 1]:
                cost_matrix[i][j] = cost_matrix[i - 1][j - 1]
            else:
                substitution = cost_matrix[i - 1][j - 1] + 1
                insertion = cost_matrix[i - 1][j] + 1
                deletion = cost_matrix[i][j - 1] + 1

                compare_val = [substitution, insertion, deletion]

                min_val = min(compare_val)
                operation_idx = compare_val.index(min_val) + 1
                cost_matrix[i][j] = min_val
                ops_matrix[i][j] = operation_idx

    match_idx = []
    i = len_hyp
    j = len_ref
    rst = {
        'nwords': len_ref,
        'cor': 0,
        'wrong': 0,
        'ins': 0,
        'del': 0,
        'sub': 0
    }
    while i >= 0 or j >= 0:
        i_idx = max(0, i)
        j_idx = max(0, j)

        if ops_matrix[i_idx][j_idx] == 0:  # correct
            if i - 1 >= 0 and j - 1 >= 0:
                match_idx.append((j - 1, i - 1))
                rst['cor'] += 1

            i -= 1
            j -= 1

        elif ops_matrix[i_idx][j_idx] == 2:  # insert
            i -= 1
            rst['ins'] += 1

        elif ops_matrix[i_idx][j_idx] == 3:  # delete
            j -= 1
            rst['del'] += 1

        elif ops_matrix[i_idx][j_idx] == 1:  # substitute
            i -= 1
            j -= 1
            rst['sub'] += 1

        if i < 0 and j >= 0:
            rst['del'] += 1
        elif j < 0 and i >= 0:
            rst['ins'] += 1

    match_idx.reverse()
    wrong_cnt = cost_matrix[len_hyp][len_ref]
    rst['wrong'] = wrong_cnt

    return rst


def print_wrong_sentence(key: str, hyp: str, ref: str):
    space = len(key)
    print(key + yellow_color + ' ref: ' + ref)
    print(' ' * space + red_color + ' hyp: ' + hyp + end_color)


def print_correct_sentence(key: str, hyp: str, ref: str):
    space = len(key)
    print(key + yellow_color + ' ref: ' + ref)
    print(' ' * space + green_color + ' hyp: ' + hyp + end_color)


def print_progress(percent):
    if percent > 1:
        percent = 1
    res = int(50 * percent) * '#'
    print('\r[%-50s] %d%%' % (res, int(100 * percent)), end='')
