import copy
import numpy as np
import time
import torch
from eend.utils.power import create_powerlabel
from itertools import combinations

metrics = [
    ('diarization_error', 'speaker_scored', 'DER'),
    ('speech_miss', 'speech_scored', 'SAD_MR'),
    ('speech_falarm', 'speech_scored', 'SAD_FR'),
    ('speaker_miss', 'speaker_scored', 'MI'),
    ('speaker_falarm', 'speaker_scored', 'FA'),
    ('speaker_error', 'speaker_scored', 'CF'),
    ('correct', 'frames', 'accuracy')
]


def recover_prediction(y, n_speaker):
    if n_speaker <= 1:
        return y
    elif n_speaker == 2:
        com_index = torch.from_numpy(
            np.array(list(combinations(np.arange(n_speaker), 2)))).to(
            y.dtype)
        num_coms = com_index.shape[0]
        y_single = y[:, :-num_coms]
        y_olp = y[:, -num_coms:]
        olp_map_index = torch.where(y_olp > 0.5)
        olp_map_index = torch.stack(olp_map_index, dim=1)
        com_map_index = com_index[olp_map_index[:, -1]]
        speaker_map_index = torch.from_numpy(np.array(com_map_index)).view(-1).to(torch.int64)
        frame_map_index = olp_map_index[:, 0][:, None].repeat([1, 2]).view(-1).to(
            torch.int64)
        y_single[frame_map_index] = 0
        y_single[frame_map_index, speaker_map_index] = 1
        return y_single
    else:
        olp2_com_index = torch.from_numpy(np.array(list(combinations(np.arange(n_speaker), 2)))).to(y.dtype)
        olp2_num_coms = olp2_com_index.shape[0]
        olp3_com_index = torch.from_numpy(np.array(list(combinations(np.arange(n_speaker), 3)))).to(y.dtype)
        olp3_num_coms = olp3_com_index.shape[0]
        y_single = y[:, :n_speaker]
        y_olp2 = y[:, n_speaker:n_speaker + olp2_num_coms]
        y_olp3 = y[:, -olp3_num_coms:]

        olp3_map_index = torch.where(y_olp3 > 0.5)
        olp3_map_index = torch.stack(olp3_map_index, dim=1)
        olp3_com_map_index = olp3_com_index[olp3_map_index[:, -1]]
        olp3_speaker_map_index = torch.from_numpy(np.array(olp3_com_map_index)).view(-1).to(torch.int64)
        olp3_frame_map_index = olp3_map_index[:, 0][:, None].repeat([1, 3]).view(-1).to(torch.int64)
        y_single[olp3_frame_map_index] = 0
        y_single[olp3_frame_map_index, olp3_speaker_map_index] = 1
        y_olp2[olp3_frame_map_index] = 0

        olp2_map_index = torch.where(y_olp2 > 0.5)
        olp2_map_index = torch.stack(olp2_map_index, dim=1)
        olp2_com_map_index = olp2_com_index[olp2_map_index[:, -1]]
        olp2_speaker_map_index = torch.from_numpy(np.array(olp2_com_map_index)).view(-1).to(torch.int64)
        olp2_frame_map_index = olp2_map_index[:, 0][:, None].repeat([1, 2]).view(-1).to(torch.int64)
        y_single[olp2_frame_map_index] = 0
        y_single[olp2_frame_map_index, olp2_speaker_map_index] = 1
        return y_single


class PowerReporter():
    def __init__(self, valid_data_loader, mapping_dict, max_n_speaker):
        valid_data_loader_cp = copy.deepcopy(valid_data_loader)
        self.valid_data_loader = valid_data_loader_cp
        del valid_data_loader
        self.mapping_dict = mapping_dict
        self.max_n_speaker = max_n_speaker

    def report(self, model, eidx, device):
        self.report_val(model, eidx, device)

    def report_val(self, model, eidx, device):
        model.eval()
        ud_valid_start = time.time()
        valid_res, valid_loss, stats_keys, vad_valid_accuracy = self.report_core(model, self.valid_data_loader, device)

        # Epoch Display
        valid_der = valid_res['diarization_error'] / valid_res['speaker_scored']
        valid_accuracy = valid_res['correct'].to(torch.float32) / valid_res['frames'] * 100
        vad_valid_accuracy = vad_valid_accuracy * 100
        print('Epoch ', eidx + 1, 'Valid Loss ', valid_loss, 'Valid_DER %.5f' % valid_der,
              'Valid_Accuracy %.5f%% ' % valid_accuracy, 'VAD_Valid_Accuracy %.5f%% ' % vad_valid_accuracy)
        ud_valid = (time.time() - ud_valid_start) / 60.
        print('Valid cost time ... ', ud_valid)

    def inv_mapping_func(self, label, mapping_dict):
        if not isinstance(label, int):
            label = int(label)
        if label in mapping_dict['label2dec'].keys():
            num = mapping_dict['label2dec'][label]
        else:
            num = -1
        return num

    def report_core(self, model, data_loader, device):
        res = {}
        for item in metrics:
            res[item[0]] = 0.
            res[item[1]] = 0.
        with torch.no_grad():
            loss_s = 0.
            uidx = 0
            for xs, ts, orders in data_loader:
                xs = [x.to(device) for x in xs]
                ts = [t.to(device) for t in ts]
                orders = [o.to(device) for o in orders]
                loss, pit_loss, mpit_loss, att_loss, ys, logits, labels, attractors = model(xs, ts, orders)
                loss_s += loss.item()
                uidx += 1

                for logit, t, att in zip(logits, labels, attractors):
                    pred = torch.argmax(torch.softmax(logit, dim=-1), dim=-1)  # (T, )
                    oov_index = torch.where(pred == self.mapping_dict['oov'])[0]
                    for i in oov_index:
                        if i > 0:
                            pred[i] = pred[i - 1]
                        else:
                            pred[i] = 0
                    pred = [self.inv_mapping_func(i, self.mapping_dict) for i in pred]
                    decisions = [bin(num)[2:].zfill(self.max_n_speaker)[::-1] for num in pred]
                    decisions = torch.from_numpy(
                        np.stack([np.array([int(i) for i in dec]) for dec in decisions], axis=0)).to(att.device).to(
                        torch.float32)
                    decisions = decisions[:, :att.shape[0]]

                    stats = self.calc_diarization_error(decisions, t)
                    res['speaker_scored'] += stats['speaker_scored']
                    res['speech_scored'] += stats['speech_scored']
                    res['frames'] += stats['frames']
                    for item in metrics:
                        res[item[0]] += stats[item[0]]
                loss_s /= uidx
                vad_acc = 0

        return res, loss_s, stats.keys(), vad_acc

    def calc_diarization_error(self, decisions, label, label_delay=0):
        label = label[:len(label) - label_delay, ...]
        n_ref = torch.sum(label, dim=-1)
        n_sys = torch.sum(decisions, dim=-1)
        res = {}
        res['speech_scored'] = torch.sum(n_ref > 0)
        res['speech_miss'] = torch.sum((n_ref > 0) & (n_sys == 0))
        res['speech_falarm'] = torch.sum((n_ref == 0) & (n_sys > 0))
        res['speaker_scored'] = torch.sum(n_ref)
        res['speaker_miss'] = torch.sum(torch.max(n_ref - n_sys, torch.zeros_like(n_ref)))
        res['speaker_falarm'] = torch.sum(torch.max(n_sys - n_ref, torch.zeros_like(n_ref)))
        n_map = torch.sum(((label == 1) & (decisions == 1)), dim=-1).to(torch.float32)
        res['speaker_error'] = torch.sum(torch.min(n_ref, n_sys) - n_map)
        res['correct'] = torch.sum(label == decisions) / label.shape[1]
        res['diarization_error'] = (
                res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
        res['frames'] = len(label)
        return res
