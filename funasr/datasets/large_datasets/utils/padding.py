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
        if data_name == "key" or data_name == "sampling_rate":
            continue
        else:
            if data_name != "hotword_indxs":
                if data[0][data_name].dtype.kind == "i":
                    pad_value = int_pad_value
                    tensor_type = torch.int64
                else:
                    pad_value = float_pad_value
                    tensor_type = torch.float32

            tensor_list = [torch.tensor(np.copy(d[data_name]), dtype=tensor_type) for d in data]
            tensor_lengths = torch.tensor([len(d[data_name]) for d in data], dtype=torch.int32)
            tensor_pad = pad_sequence(tensor_list, batch_first=True, padding_value=pad_value)
            batch[data_name] = tensor_pad
            batch[data_name + "_lengths"] = tensor_lengths

    # SAC LABEL INCLUDE
    if "hotword_indxs" in batch:
        # if hotword indxs in batch
        # use it to slice hotwords out
        hotword_list = []
        hotword_lengths = []
        text = batch["text"]
        text_lengths = batch["text_lengths"]
        hotword_indxs = batch["hotword_indxs"]
        dha_pad = torch.ones_like(text) * -1
        _, t1 = text.shape
        t1 += 1  # TODO: as parameter which is same as predictor_bias
        nth_hw = 0
        for b, (hotword_indx, one_text, length) in enumerate(
            zip(hotword_indxs, text, text_lengths)
        ):
            dha_pad[b][:length] = 8405
            if hotword_indx[0] != -1:
                start, end = int(hotword_indx[0]), int(hotword_indx[1])
                hotword = one_text[start : end + 1]
                hotword_list.append(hotword)
                hotword_lengths.append(end - start + 1)
                dha_pad[b][start : end + 1] = one_text[start : end + 1]
                nth_hw += 1
                if len(hotword_indx) == 4 and hotword_indx[2] != -1:
                    # the second hotword if exist
                    start, end = int(hotword_indx[2]), int(hotword_indx[3])
                    hotword_list.append(one_text[start : end + 1])
                    hotword_lengths.append(end - start + 1)
                    dha_pad[b][start : end + 1] = one_text[start : end + 1]
                    nth_hw += 1
        hotword_list.append(torch.tensor([1]))
        hotword_lengths.append(1)
        hotword_pad = pad_sequence(hotword_list, batch_first=True, padding_value=0)
        batch["hotword_pad"] = hotword_pad
        batch["hotword_lengths"] = torch.tensor(hotword_lengths, dtype=torch.int32)
        batch["dha_pad"] = dha_pad
        del batch["hotword_indxs"]
        del batch["hotword_indxs_lengths"]
    return keys, batch
