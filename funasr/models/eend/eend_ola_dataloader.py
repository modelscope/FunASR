import logging

import kaldiio
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def custom_collate(batch):
    keys, speech, speaker_labels, orders = zip(*batch)
    speech = [torch.from_numpy(np.copy(sph)).to(torch.float32) for sph in speech]
    speaker_labels = [torch.from_numpy(np.copy(spk)).to(torch.float32) for spk in speaker_labels]
    orders = [torch.from_numpy(np.copy(o)).to(torch.int64) for o in orders]
    batch = dict(speech=speech, speaker_labels=speaker_labels, orders=orders)

    return keys, batch


class EENDOLADataset(Dataset):
    def __init__(
        self,
        data_file,
    ):
        self.data_file = data_file
        with open(data_file) as f:
            lines = f.readlines()
        self.samples = [line.strip().split() for line in lines]
        logging.info("total samples: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key, speech_path, speaker_label_path = self.samples[idx]
        speech = kaldiio.load_mat(speech_path)
        speaker_label = kaldiio.load_mat(speaker_label_path).reshape(speech.shape[0], -1)

        order = np.arange(speech.shape[0])
        np.random.shuffle(order)

        return key, speech, speaker_label, order


class EENDOLADataLoader:
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=8):
        dataset = EENDOLADataset(data_file)
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=custom_collate,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def build_iter(self, epoch):
        return self.data_loader
