import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def padding(data, float_pad_value=0.0, int_pad_value=-1):
    assert isinstance(data, list)
    assert "key" in data[0]
    assert "speech" in data[0] or "text" in data[0]
    
    keys = [x["key"] for x in data]

    batch = {}
    data_names = data[0].keys()
    for data_name in data_names:
        if data_name == "key" or data_name == "sampling_rate" or data_name == 'hotword_indxs':
            continue
        else:
            if data[0][data_name].dtype.kind == "i":
                pad_value = int_pad_value
                tensor_type = torch.int64
            else:
                pad_value = float_pad_value
                tensor_type = torch.float32

            tensor_list = [torch.tensor(np.copy(d[data_name]), dtype=tensor_type) for d in data]
            tensor_lengths = torch.tensor([len(d[data_name]) for d in data], dtype=torch.int32)
            tensor_pad = pad_sequence(tensor_list,
                                      batch_first=True,
                                      padding_value=pad_value)
            batch[data_name] = tensor_pad
            batch[data_name + "_lengths"] = tensor_lengths

    # DHA, EAHC NOT INCLUDED
    if "hotword_indxs" in batch:
        # if hotword indxs in batch
        # use it to slice hotwords out
        hotword_list = []
        hotword_lengths = []
        text = batch['text']
        text_lengths = batch['text_lengths']
        hotword_indxs = batch['hotword_indxs']
        num_hw = sum([int(i) for i in batch['hotword_indxs_lengths'] if i != 1]) // 2
        B, t1 = text.shape
        t1 += 1  # TODO: as parameter which is same as predictor_bias
        ideal_attn = torch.zeros(B, t1, num_hw+1)
        nth_hw = 0
        for b, (hotword_indx, one_text, length) in enumerate(zip(hotword_indxs, text, text_lengths)):
            ideal_attn[b][:,-1] = 1
            if hotword_indx[0] != -1:
                start, end = int(hotword_indx[0]), int(hotword_indx[1])
                hotword = one_text[start: end+1]
                hotword_list.append(hotword)
                hotword_lengths.append(end-start+1)
                ideal_attn[b][start:end+1, nth_hw] = 1
                ideal_attn[b][start:end+1, -1] = 0
                nth_hw += 1
                if len(hotword_indx) == 4 and hotword_indx[2] != -1:
                    # the second hotword if exist
                    start, end = int(hotword_indx[2]), int(hotword_indx[3])
                    hotword_list.append(one_text[start: end+1])
                    hotword_lengths.append(end-start+1)
                    ideal_attn[b][start:end+1, nth_hw-1] = 1
                    ideal_attn[b][start:end+1, -1] = 0
                    nth_hw += 1
        hotword_list.append(torch.tensor([1]))
        hotword_lengths.append(1)
        hotword_pad = pad_sequence(hotword_list,
                                batch_first=True,
                                padding_value=0)
        batch["hotword_pad"] = hotword_pad
        batch["hotword_lengths"] = torch.tensor(hotword_lengths, dtype=torch.int32)
        batch['ideal_attn'] = ideal_attn
        del batch['hotword_indxs']
        del batch['hotword_indxs_lengths']

    return keys, batch
