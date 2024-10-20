""" This implementation is adapted from https://github.com/wenet-e2e/wekws/blob/main/wekws/bin/compute_det.py."""

import os
import json
import logging
import argparse
import threading

import kaldiio
import torch
from funasr.utils.kws_utils import split_mixed_label


class thread_wrapper(threading.Thread):
    def __init__(self, func, args=()):
        super(thread_wrapper, self).__init__()
        self.func = func
        self.args = args
        self.result = []

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def space_mixed_label(input_str):
    splits = split_mixed_label(input_str)
    space_str = ''.join(f'{sub} ' for sub in splits)
    return space_str.strip()


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            if line.strip() != '':
                lists.append(line.strip())
    return lists


def make_pair(wav_lists, trans_lists):
    logging.info('make pair for wav-trans list')

    trans_table = {}
    for line in trans_lists:
        arr = line.strip().replace('\t', ' ').split()
        if len(arr) < 2:
            logging.debug('invalid line in trans file: {}'.format(
                line.strip()))
            continue

        trans_table[arr[0]] = line.replace(arr[0],'').strip()

    lists = []
    for line in wav_lists:
        arr = line.strip().replace('\t', ' ').split()
        if len(arr) == 2 and arr[0] in trans_table:
            lists.append(
                dict(key=arr[0],
                     txt=trans_table[arr[0]],
                     wav=arr[1],
                     sample_rate=16000))
        else:
            logging.debug("can't find corresponding trans for key: {}".format(
                arr[0]))
            continue

    return lists


def count_duration(tid, data_lists):
    results = []

    for obj in data_lists:
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']

        try:
            rate, waveform = kaldiio.load_mat(wav_file)
            waveform = torch.tensor(waveform, dtype=torch.float32)
            waveform = waveform.unsqueeze(0)
            frames = len(waveform[0])
            duration = frames / float(rate)
        except:
            logging.info(f'load file failed: {wav_file}')
            duration = 0.0

        obj['duration'] = duration
        results.append(obj)

    return results


def load_data_and_score(keywords_list, data_file, trans_file, score_file):
    # score_table: {uttid: [keywordlist]}
    score_table = {}
    with open(score_file, 'r', encoding='utf8') as fin:
        # read score file and store in table
        for line in fin:
            arr = line.strip().split()
            key = arr[0]
            is_detected = arr[1]
            if is_detected == 'detected':
                if key not in score_table:
                    score_table.update(
                        {key: {
                            'kw': space_mixed_label(arr[2]),
                            'confi': float(arr[3])
                        }})
            else:
                if key not in score_table:
                    score_table.update({key: {'kw': 'unknown', 'confi': -1.0}})

    wav_lists = read_lists(data_file)
    trans_lists = read_lists(trans_file)
    data_lists = make_pair(wav_lists, trans_lists)
    logging.info(f'origin list samples: {len(data_lists)}')

    # count duration for each wave
    num_workers = 8
    start = 0
    step = int(len(data_lists) / num_workers)
    tasks = []
    for idx in range(num_workers):
        if idx != num_workers - 1:
            task = thread_wrapper(count_duration,
                                  (idx, data_lists[start:start + step]))
        else:
            task = thread_wrapper(count_duration, (idx, data_lists[start:]))
        task.start()
        tasks.append(task)
        start += step

    duration_lists = []
    for task in tasks:
        task.join()
        duration_lists += task.get_result()
    logging.info(f'after list samples: {len(duration_lists)}')

    # build empty structure for keyword-filler infos
    keyword_filler_table = {}
    for keyword in keywords_list:
        keyword = space_mixed_label(keyword)
        keyword_filler_table[keyword] = {}
        keyword_filler_table[keyword]['keyword_table'] = {}
        keyword_filler_table[keyword]['keyword_duration'] = 0.0
        keyword_filler_table[keyword]['filler_table'] = {}
        keyword_filler_table[keyword]['filler_duration'] = 0.0

    for obj in duration_lists:
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        assert 'duration' in obj

        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']
        txt = space_mixed_label(txt)
        txt_regstr_lrblk = ' ' + txt + ' '
        duration = obj['duration']
        assert key in score_table

        for keyword in keywords_list:
            keyword = space_mixed_label(keyword)
            keyword_regstr_lrblk = ' ' + keyword + ' '
            if txt_regstr_lrblk.find(keyword_regstr_lrblk) != -1:
                if keyword == score_table[key]['kw']:
                    keyword_filler_table[keyword]['keyword_table'].update(
                        {key: score_table[key]['confi']})
                else:
                    # uttrance detected but not match this keyword
                    keyword_filler_table[keyword]['keyword_table'].update(
                        {key: -1.0})
                keyword_filler_table[keyword]['keyword_duration'] += duration
            else:
                if keyword == score_table[key]['kw']:
                    keyword_filler_table[keyword]['filler_table'].update(
                        {key: score_table[key]['confi']})
                else:
                    # uttrance if detected, which is not FA for this keyword
                    keyword_filler_table[keyword]['filler_table'].update(
                        {key: -1.0})
                keyword_filler_table[keyword]['filler_duration'] += duration

    return keyword_filler_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute det curve')
    parser.add_argument('--keywords',
                        type=str,
                        required=True,
                        help='preset keyword str, input all keywords')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--trans_data',
                        required=True,
                        default='',
                        help='transcription of test data')
    parser.add_argument('--score_file', required=True, help='score file')
    parser.add_argument('--step',
                        type=float,
                        default=0.001,
                        help='threshold step')
    parser.add_argument('--stats_dir',
                        required=True,
                        help='to save det stats files')
    args = parser.parse_args()

    root_logger = logging.getLogger()
    handlers = root_logger.handlers[:]
    for handler in handlers:
        root_logger.removeHandler(handler)
        handler.close()

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    keywords_list = args.keywords.strip().split(',')
    keyword_filler_table = load_data_and_score(keywords_list, args.test_data,
                                               args.trans_data,
                                               args.score_file)

    stats_files = {}
    for keyword in keywords_list:
        keyword = space_mixed_label(keyword)
        keyword_dur = keyword_filler_table[keyword]['keyword_duration']
        keyword_num = len(keyword_filler_table[keyword]['keyword_table'])
        filler_dur = keyword_filler_table[keyword]['filler_duration']
        filler_num = len(keyword_filler_table[keyword]['filler_table'])
        if keyword_num <= 0:
            print('Can\'t compute det for {} without positive sample'.format(keyword))
            continue
        if filler_num <= 0:
            print('Can\'t compute det for {} without negative sample'.format(keyword))
            continue

        logging.info('Computing det for {}'.format(keyword))
        logging.info('  Keyword duration: {} Hours, wave number: {}'.format(
            keyword_dur / 3600.0, keyword_num))
        logging.info('  Filler duration: {} Hours'.format(filler_dur / 3600.0))

        stats_file = os.path.join(args.stats_dir, 'stats.' + keyword.replace(' ', '_') + '.txt')
        with open(stats_file, 'w', encoding='utf8') as fout:
            threshold = 0.0
            while threshold <= 1.0:
                num_false_reject = 0
                num_true_detect = 0
                # transverse the all keyword_table
                for key, confi in keyword_filler_table[keyword][
                        'keyword_table'].items():
                    if confi < threshold:
                        num_false_reject += 1
                    else:
                        num_true_detect += 1

                num_false_alarm = 0
                # transverse the all filler_table
                for key, confi in keyword_filler_table[keyword][
                        'filler_table'].items():
                    if confi >= threshold:
                        num_false_alarm += 1
                        # print(f'false alarm: {keyword}, {key}, {confi}')

                # false_reject_rate = num_false_reject / keyword_num
                true_detect_rate = num_true_detect / keyword_num

                num_false_alarm = max(num_false_alarm, 1e-6)
                false_alarm_per_hour = num_false_alarm / (filler_dur / 3600.0)
                false_alarm_rate = num_false_alarm / filler_num

                fout.write('{:.3f} {:.6f} {:.6f} {:.6f}\n'.format(
                    threshold, true_detect_rate, false_alarm_rate,
                    false_alarm_per_hour))
                threshold += args.step

        stats_files[keyword] = stats_file
