from torch.utils.data import IterableDataset


def default_fn(data):
    return data


class FilterIterDataPipe(IterableDataset):

    def __init__(self, datapipe, fn=default_fn):
        self.datapipe = datapipe
        self.fn = fn

    def set_epoch(self, epoch):
        self.datapipe.set_epoch(epoch)

    def __iter__(self):
        assert callable(self.fn)
        for data in self.datapipe:
            if self.fn(data):
                yield data
            else:
                continue
