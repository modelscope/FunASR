import random

from itertools import count
from functools import partial
from torch.utils.data import IterableDataset
from funasr.datasets.large_datasets.datapipes.map import MapperIterDataPipe

tiebreaker = count()


def _default_len_fn(token):
    return len(token), next(tiebreaker)


def _token_len_fn(token, len_fn):
    return len_fn(token), next(tiebreaker), token


class MaxTokenBucketizerIterDataPipe(IterableDataset):

    def __init__(
        self,
        datapipe,
        batch_size=8000,
        len_fn=_default_len_fn,
        buffer_size=10240,
        sort_size=500,
        batch_mode="padding",
    ):
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        assert buffer_size >= -1, "Buffer size is required to be larger than -1!"
        assert sort_size > 0, "Sort size is required to be larger than 0!"

        datapipe = MapperIterDataPipe(datapipe, fn=partial(_token_len_fn, len_fn=len_fn))
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.sort_size = sort_size
        self.batch_mode = batch_mode

    def set_epoch(self, epoch):
        self.datapipe.set_epoch(epoch)

    def __iter__(self):
        buffer = []
        batch = []
        bucket = []
        max_lengths = 0
        min_lengths = 999999
        batch_lengths = 0

        if self.batch_mode == "clipping":
            assert self.buffer_size > 0, "for clipping batch_mode, buffer_size must be > 1"
            for d in self.datapipe:
                if d[0] > self.batch_size:
                    continue
                buffer.append(d)
                if len(buffer) == self.buffer_size:
                    random.shuffle(buffer)
                    for sample in buffer:
                        bucket.append(sample)
                        if len(bucket) == self.sort_size:
                            bucket.sort()
                            for x in bucket:
                                length, _, token = x
                                if length < min_lengths:
                                    min_lengths = length
                                batch_lengths = min_lengths * (len(batch) + 1)
                                if batch_lengths > self.batch_size:
                                    yield batch
                                    batch = []
                                    min_lengths = length
                                batch.append(token)
                            bucket = []
                    buffer = []

            if buffer:
                random.shuffle(buffer)
                for sample in buffer:
                    bucket.append(sample)
                    if len(bucket) == self.sort_size:
                        bucket.sort()
                        for x in bucket:
                            length, _, token = x
                            if length < min_lengths:
                                min_lengths = length
                            batch_lengths = min_lengths * (len(batch) + 1)
                            if batch_lengths > self.batch_size:
                                yield batch
                                batch = []
                                min_lengths = length
                            batch.append(token)
                        bucket = []
                buffer = []

            if bucket:
                bucket.sort()
                for x in bucket:
                    length, _, token = x
                    if length < min_lengths:
                        min_lengths = length
                    batch_lengths = min_lengths * (len(batch) + 1)
                    if batch_lengths > self.batch_size:
                        yield batch
                        batch = []
                        min_lengths = length
                    batch.append(token)
                bucket = []

            if batch:
                yield batch

        else:
            if self.buffer_size == -1:
                for d in self.datapipe:
                    if d[0] > self.batch_size:
                        continue
                    buffer.append(d)
                buffer.sort()
                for sample in buffer:
                    length, _, token = sample
                    if length > max_lengths:
                        max_lengths = length
                    batch_lengths = max_lengths * (len(batch) + 1)
                    if batch_lengths > self.batch_size:
                        bucket.append(batch)
                        batch = []
                        max_lengths = length
                    batch.append(token)
                random.shuffle(bucket)
                if bucket:
                    for batch_sample in bucket:
                        yield batch_sample
                if batch:
                    yield batch

            elif self.buffer_size == 0:
                for d in self.datapipe:
                    if d[0] > self.batch_size:
                        continue
                    length, _, token = d
                    if length > self.batch_size:
                        continue
                    if length > max_lengths:
                        max_lengths = length
                    batch_lengths = max_lengths * (len(batch) + 1)
                    if batch_lengths > self.batch_size:
                        yield batch
                        batch = []
                        max_lengths = length
                    batch.append(token)
                if batch:
                    yield batch

            else:
                for d in self.datapipe:
                    if d[0] > self.batch_size:
                        continue
                    buffer.append(d)
                    if len(buffer) == self.buffer_size:
                        random.shuffle(buffer)
                        for sample in buffer:
                            bucket.append(sample)
                            if len(bucket) == self.sort_size:
                                bucket.sort()
                                for x in bucket:
                                    length, _, token = x
                                    if length > max_lengths:
                                        max_lengths = length
                                    batch_lengths = max_lengths * (len(batch) + 1)
                                    if batch_lengths > self.batch_size:
                                        yield batch
                                        batch = []
                                        max_lengths = length
                                    batch.append(token)
                                bucket = []
                        buffer = []

                if buffer:
                    random.shuffle(buffer)
                    for sample in buffer:
                        bucket.append(sample)
                        if len(bucket) == self.sort_size:
                            bucket.sort()
                            for x in bucket:
                                length, _, token = x
                                if length > max_lengths:
                                    max_lengths = length
                                batch_lengths = max_lengths * (len(batch) + 1)
                                if batch_lengths > self.batch_size:
                                    yield batch
                                    batch = []
                                    max_lengths = length
                                batch.append(token)
                            bucket = []
                    buffer = []

                if bucket:
                    bucket.sort()
                    for x in bucket:
                        length, _, token = x
                        if length > max_lengths:
                            max_lengths = length
                        batch_lengths = max_lengths * (len(batch) + 1)
                        if batch_lengths > self.batch_size:
                            yield batch
                            batch = []
                            max_lengths = length
                        batch.append(token)
                    bucket = []

                if batch:
                    yield batch
