import unittest

from funasr.datasets.audio_datasets.samplers import (
    CustomDistributedBatchSampler,
    CustomDistributedBufferBatchSampler,
)


class _Dataset:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def get_source_len(self, index):
        return 1


class TestBatchSamplerDropLast(unittest.TestCase):
    def test_drop_last_keeps_only_full_batches(self):
        sampler = CustomDistributedBatchSampler(
            _Dataset(10), batch_size=4, drop_last=True, shuffle=False
        )
        batches = list(sampler)
        self.assertEqual(batches, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(len(sampler), len(batches))

    def test_drop_last_with_shuffle_never_repeats_an_example(self):
        sampler = CustomDistributedBatchSampler(
            _Dataset(10), batch_size=4, drop_last=True, shuffle=True
        )
        seen = [i for batch in sampler for i in batch]
        self.assertEqual(len(seen), 8)
        self.assertEqual(len(set(seen)), 8)

    def test_drop_last_with_fewer_examples_than_one_batch_yields_nothing(self):
        sampler = CustomDistributedBatchSampler(
            _Dataset(3), batch_size=4, drop_last=True, shuffle=False
        )
        self.assertEqual(list(sampler), [])

    def test_without_drop_last_the_last_batch_is_padded(self):
        sampler = CustomDistributedBatchSampler(
            _Dataset(10), batch_size=4, drop_last=False, shuffle=False
        )
        self.assertEqual(list(sampler), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]])

    def test_buffer_sampler_drop_last_keeps_only_full_batches(self):
        sampler = CustomDistributedBufferBatchSampler(
            _Dataset(10), batch_size=4, drop_last=True, shuffle=False, sort_size=1024
        )
        self.assertEqual(list(sampler), [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_buffer_sampler_drop_last_with_fewer_examples_than_one_batch(self):
        sampler = CustomDistributedBufferBatchSampler(
            _Dataset(3), batch_size=4, drop_last=True, shuffle=False, sort_size=1024
        )
        self.assertEqual(list(sampler), [])


if __name__ == "__main__":
    unittest.main()
