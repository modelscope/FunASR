import torch
import codecs
import logging
import argparse
import numpy as np
import edit_distance
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
        integrate = torch.where(fire_place,
                                integrate - torch.ones([batch_size], device=alphas.device),
                                integrate)
    fires = torch.stack(list_fires, 1)
    return fires


def ts_prediction_lfr6_standard(us_alphas, 
                       us_peaks, 
                       char_list, 
                       vad_offset=0.0, 
                       force_time_shift=-1.5,
                       sil_in_str=True
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
    if char_list[-1] == '</s>':
        char_list = char_list[:-1]
    fire_place = torch.where(peaks>1.0-1e-4)[0].cpu().numpy() + force_time_shift  # total offset
    if len(fire_place) != len(char_list) + 1:
        alphas /= (alphas.sum() / (len(char_list) + 1))
        alphas = alphas.unsqueeze(0)
        peaks = cif_wo_hidden(alphas, threshold=1.0-1e-4)[0]
        fire_place = torch.where(peaks>1.0-1e-4)[0].cpu().numpy() + force_time_shift  # total offset
    num_frames = peaks.shape[0]
    timestamp_list = []
    new_char_list = []
    # for bicif model trained with large data, cif2 actually fires when a character starts
    # so treat the frames between two peaks as the duration of the former token
    fire_place = torch.where(peaks>1.0-1e-4)[0].cpu().numpy() + force_time_shift  # total offset
    # assert num_peak == len(char_list) + 1 # number of peaks is supposed to be number of tokens + 1
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
        #if char != '<sil>':
        if not sil_in_str and char == '<sil>': continue
        res_txt += "{} {} {};".format(char, str(timestamp[0]+0.0005)[:5], str(timestamp[1]+0.0005)[:5])
    res = []
    for char, timestamp in zip(new_char_list, timestamp_list):
        if char != '<sil>':
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res_txt, res


def time_stamp_sentence(punc_id_list, time_stamp_postprocessed, text_postprocessed):
    punc_list = ['，', '。', '？', '、']
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
            "end": time_stamp_postprocessed[-1][1],
            'text_seg': text_postprocessed.split(),
            "ts_list": time_stamp_postprocessed,
        })
        return res
    if len(punc_id_list) != len(time_stamp_postprocessed):
        print("  warning length mistach!!!!!!")
    sentence_text = ""
    sentence_text_seg = ""
    ts_list = []
    sentence_start = time_stamp_postprocessed[0][0]
    sentence_end = time_stamp_postprocessed[0][1]
    texts = text_postprocessed.split()
    punc_stamp_text_list = list(zip_longest(punc_id_list, time_stamp_postprocessed, texts, fillvalue=None))
    for punc_stamp_text in punc_stamp_text_list:
        punc_id, time_stamp, text = punc_stamp_text
        # sentence_text += text if text is not None else ''
        if text is not None:
            if 'a' <= text[0] <= 'z' or 'A' <= text[0] <= 'Z':
                sentence_text += ' ' + text
            elif len(sentence_text) and ('a' <= sentence_text[-1] <= 'z' or 'A' <= sentence_text[-1] <= 'Z'):
                sentence_text += ' ' + text
            else:
                sentence_text += text
            sentence_text_seg += text + ' '
        ts_list.append(time_stamp)

        punc_id = int(punc_id) if punc_id is not None else 1
        sentence_end = time_stamp[1] if time_stamp is not None else sentence_end

        if punc_id > 1:
            sentence_text += punc_list[punc_id - 2]
            res.append({
                'text': sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "text_seg": sentence_text_seg,
                "ts_list": ts_list
            })
            sentence_text = ''
            sentence_text_seg = ''
            ts_list = []
            sentence_start = sentence_end
    return res


class AverageShiftCalculator():
    def __init__(self):
        logging.warning("Calculating average shift.")
    def __call__(self, file1, file2):
        uttid_list1, ts_dict1 = self.read_timestamps(file1)
        uttid_list2, ts_dict2 = self.read_timestamps(file2)
        uttid_intersection = self._intersection(uttid_list1, uttid_list2)
        res = self.as_cal(uttid_intersection, ts_dict1, ts_dict2)
        logging.warning("Average shift of {} and {}: {}.".format(file1, file2, str(res)[:8]))
        logging.warning("Following timestamp pair differs most: {}, detail:{}".format(self.max_shift, self.max_shift_uttid))

    def _intersection(self, list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        if set1 == set2:
            logging.warning("Uttid same checked.")
            return set1
        itsc = list(set1 & set2)
        logging.warning("Uttid differs: file1 {}, file2 {}, lines same {}.".format(len(list1), len(list2), len(itsc)))
        return itsc

    def read_timestamps(self, file):
        # read timestamps file in standard format
        uttid_list = []
        ts_dict = {}
        with codecs.open(file, 'r') as fin:
            for line in fin.readlines():
                text = ''
                ts_list = []
                line = line.rstrip()
                uttid = line.split()[0]
                uttid_list.append(uttid)
                body = " ".join(line.split()[1:])
                for pd in body.split(';'):
                    if not len(pd): continue
                    # pdb.set_trace() 
                    char, start, end = pd.lstrip(" ").split(' ')
                    text += char + ','
                    ts_list.append((float(start), float(end)))
                # ts_lists.append(ts_list)
                ts_dict[uttid] = (text[:-1], ts_list)
        logging.warning("File {} read done.".format(file))
        return uttid_list, ts_dict

    def _shift(self, filtered_timestamp_list1, filtered_timestamp_list2):
        shift_time = 0
        for fts1, fts2 in zip(filtered_timestamp_list1, filtered_timestamp_list2):
            shift_time += abs(fts1[0] - fts2[0]) + abs(fts1[1] - fts2[1])
        num_tokens = len(filtered_timestamp_list1)
        return shift_time, num_tokens

    def as_cal(self, uttid_list, ts_dict1, ts_dict2):
        # calculate average shift between timestamp1 and timestamp2
        # when characters differ, use edit distance alignment
        # and calculate the error between the same characters
        self._accumlated_shift = 0
        self._accumlated_tokens = 0
        self.max_shift = 0
        self.max_shift_uttid = None
        for uttid in uttid_list:
            (t1, ts1) = ts_dict1[uttid]
            (t2, ts2) = ts_dict2[uttid]
            _align, _align2, _align3 = [], [], []
            fts1, fts2 = [], []
            _t1, _t2 = [], []
            sm = edit_distance.SequenceMatcher(t1.split(','), t2.split(','))
            s = sm.get_opcodes()
            for j in range(len(s)):
                if s[j][0] == "replace" or s[j][0] == "insert":
                    _align.append(0)
                if s[j][0] == "replace" or s[j][0] == "delete":
                    _align3.append(0)
                elif s[j][0] == "equal":
                    _align.append(1)
                    _align3.append(1)
                else:
                    continue
            # use s to index t2
            for a, ts , t in zip(_align, ts2, t2.split(',')):
                if a: 
                    fts2.append(ts)
                    _t2.append(t)
            sm2 = edit_distance.SequenceMatcher(t2.split(','), t1.split(','))
            s = sm2.get_opcodes()
            for j in range(len(s)):
                if s[j][0] == "replace" or s[j][0] == "insert":
                    _align2.append(0)
                elif s[j][0] == "equal":
                    _align2.append(1)
                else:
                    continue
            # use s2 tp index t1
            for a, ts, t in zip(_align3, ts1, t1.split(',')):
                if a: 
                    fts1.append(ts)
                    _t1.append(t)
            if len(fts1) == len(fts2):
                shift_time, num_tokens = self._shift(fts1, fts2)
                self._accumlated_shift += shift_time
                self._accumlated_tokens += num_tokens
                if shift_time/num_tokens > self.max_shift:
                    self.max_shift = shift_time/num_tokens
                    self.max_shift_uttid = uttid
            else:
                logging.warning("length mismatch")
        return self._accumlated_shift / self._accumlated_tokens


def convert_external_alphas(alphas_file, text_file, output_file):
    from funasr.models.predictor.cif import cif_wo_hidden
    with open(alphas_file, 'r') as f1, open(text_file, 'r') as f2, open(output_file, 'w') as f3:
        for line1, line2 in zip(f1.readlines(), f2.readlines()):
            line1 = line1.rstrip()
            line2 = line2.rstrip()
            assert line1.split()[0] == line2.split()[0]
            uttid = line1.split()[0]
            alphas = [float(i) for i in line1.split()[1:]]
            new_alphas = np.array(remove_chunk_padding(alphas))
            new_alphas[-1] += 1e-4
            text = line2.split()[1:]
            if len(text) + 1 != int(new_alphas.sum()):
                # force resize
                new_alphas *= (len(text) + 1) / int(new_alphas.sum())
            peaks = cif_wo_hidden(torch.Tensor(new_alphas).unsqueeze(0), 1.0-1e-4)
            if " " in text:
                text = text.split()
            else:
                text = [i for i in text]
            res_str, _ = ts_prediction_lfr6_standard(new_alphas, peaks[0], text, 
                                                     force_time_shift=-7.0, 
                                                     sil_in_str=False)
            f3.write("{} {}\n".format(uttid, res_str))


def remove_chunk_padding(alphas):
    # remove the padding part in alphas if using chunk paraformer for GPU
    START_ZERO = 45
    MID_ZERO = 75
    REAL_FRAMES = 360  # for chunk based encoder 10-120-10 and fsmn padding 5
    alphas = alphas[START_ZERO:]  # remove the padding at beginning
    new_alphas = []
    while True:
        new_alphas = new_alphas + alphas[:REAL_FRAMES]
        alphas = alphas[REAL_FRAMES+MID_ZERO:]
        if len(alphas) < REAL_FRAMES: break
    return new_alphas

SUPPORTED_MODES = ['cal_aas', 'read_ext_alphas']


def main(args):
    if args.mode == 'cal_aas':
        asc = AverageShiftCalculator()
        asc(args.input, args.input2)
    elif args.mode == 'read_ext_alphas':
        convert_external_alphas(args.input, args.input2, args.output)
    else:
        logging.error("Mode {} not in SUPPORTED_MODES: {}.".format(args.mode, SUPPORTED_MODES))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='timestamp tools')
    parser.add_argument('--mode', 
                        default=None, 
                        type=str, 
                        choices=SUPPORTED_MODES, 
                        help='timestamp related toolbox')
    parser.add_argument('--input', default=None, type=str, help='input file path')
    parser.add_argument('--output', default=None, type=str, help='output file name')
    parser.add_argument('--input2', default=None, type=str, help='input2 file path')
    parser.add_argument('--kaldi-ts-type', 
                        default='v2', 
                        type=str, 
                        choices=['v0', 'v1', 'v2'], 
                        help='kaldi timestamp to write')
    args = parser.parse_args()
    main(args)

