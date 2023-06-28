import logging
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch

from funasr.iterators.abs_iter_factory import AbsIterFactory
from funasr.iterators.sequence_iter_factory import SequenceIterFactory
from funasr.samplers.abs_sampler import AbsSampler


class ChunkIterFactory(AbsIterFactory):
    """Creates chunks from a sequence

    Examples:
        >>> batches = [["id1"], ["id2"], ...]
        >>> batch_size = 128
        >>> chunk_length = 1000
        >>> iter_factory = ChunkIterFactory(dataset, batches, batch_size, chunk_length)
        >>> it = iter_factory.build_iter(epoch)
        >>> for ids, batch in it:
        ...     ...

    - The number of mini-batches are varied in each epochs and
      we can't get the number in advance
      because IterFactory doesn't be given to the length information.
    - Since the first reason, "num_iters_per_epoch" can't be implemented
      for this iterator. Instead of it, "num_samples_per_epoch" is implemented.

    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        batches: Union[AbsSampler, Sequence[Sequence[Any]]],
        chunk_length: Union[int, str],
        chunk_shift_ratio: float = 0.5,
        num_cache_chunks: int = 1024,
        num_samples_per_epoch: int = None,
        seed: int = 0,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
    ):
        assert check_argument_types()
        assert all(len(x) == 1 for x in batches), "batch-size must be 1"

        self.per_sample_iter_factory = SequenceIterFactory(
            dataset=dataset,
            batches=batches,
            num_iters_per_epoch=num_samples_per_epoch,
            seed=seed,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

        self.num_cache_chunks = max(num_cache_chunks, batch_size)
        if isinstance(chunk_length, str):
            if len(chunk_length) == 0:
                raise ValueError("e.g. 5,8 or 3-5: but got empty string")

            self.chunk_lengths = []
            for x in chunk_length.split(","):
                try:
                    sps = list(map(int, x.split("-")))
                except ValueError:
                    raise ValueError(f"e.g. 5,8 or 3-5: but got {chunk_length}")

                if len(sps) > 2:
                    raise ValueError(f"e.g. 5,8 or 3-5: but got {chunk_length}")
                elif len(sps) == 2:
                    # Append all numbers between the range into the candidates
                    self.chunk_lengths += list(range(sps[0], sps[1] + 1))
                else:
                    self.chunk_lengths += [sps[0]]
        else:
            # Single candidates: Fixed chunk length
            self.chunk_lengths = [chunk_length]

        self.chunk_shift_ratio = chunk_shift_ratio
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle

    def build_iter(
        self,
        epoch: int,
        shuffle: bool = None,
    ) -> Iterator[Tuple[List[str], Dict[str, torch.Tensor]]]:
        per_sample_loader = self.per_sample_iter_factory.build_iter(epoch, shuffle)

        if shuffle is None:
            shuffle = self.shuffle
        state = np.random.RandomState(epoch + self.seed)

        # NOTE(kamo):
        #   This iterator supports multiple chunk lengths and
        #   keep chunks for each lengths here until collecting specified numbers
        cache_chunks_dict = {}
        cache_id_list_dict = {}
        for ids, batch in per_sample_loader:
            # Must be per-sample-loader
            assert len(ids) == 1, f"Must be per-sample-loader: {len(ids)}"
            assert all(len(x) == 1 for x in batch.values())

            # Get keys of sequence data
            sequence_keys = []
            for key in batch:
                if key + "_lengths" in batch:
                    sequence_keys.append(key)
            # Remove lengths data and get the first sample
            batch = {k: v[0] for k, v in batch.items() if not k.endswith("_lengths")}
            id_ = ids[0]

            for key in sequence_keys:
                if len(batch[key]) != len(batch[sequence_keys[0]]):
                    raise RuntimeError(
                        f"All sequences must has same length: "
                        f"{len(batch[key])} != {len(batch[sequence_keys[0]])}"
                    )

            L = len(batch[sequence_keys[0]])
            # Select chunk length
            chunk_lengths = [lg for lg in self.chunk_lengths if lg < L]
            if len(chunk_lengths) == 0:
                logging.warning(
                    f"The length of '{id_}' is {L}, but it is shorter than "
                    f"any candidates of chunk-length: {self.chunk_lengths}"
                )
                continue

            W = int(state.choice(chunk_lengths, 1))
            cache_id_list = cache_id_list_dict.setdefault(W, [])
            cache_chunks = cache_chunks_dict.setdefault(W, {})

            # Shift width to the next chunk
            S = int(W * self.chunk_shift_ratio)
            # Number of chunks
            N = (L - W) // S + 1
            if shuffle:
                Z = state.randint(0, (L - W) % S + 1)
            else:
                Z = 0

            # Split a sequence into chunks.
            # Note that the marginal frames divided by chunk length are discarded
            for k, v in batch.items():
                if k not in cache_chunks:
                    cache_chunks[k] = []
                if k in sequence_keys:
                    # Shift chunks with overlapped length for data augmentation
                    cache_chunks[k] += [v[Z + i * S : Z + i * S + W] for i in range(N)]
                else:
                    # If not sequence, use whole data instead of chunk
                    cache_chunks[k] += [v for _ in range(N)]
            cache_id_list += [id_ for _ in range(N)]

            if len(cache_id_list) > self.num_cache_chunks:
                cache_id_list, cache_chunks = yield from self._generate_mini_batches(
                    cache_id_list,
                    cache_chunks,
                    shuffle,
                    state,
                )

            cache_id_list_dict[W] = cache_id_list
            cache_chunks_dict[W] = cache_chunks

        else:
            for W in cache_id_list_dict:
                cache_id_list = cache_id_list_dict.setdefault(W, [])
                cache_chunks = cache_chunks_dict.setdefault(W, {})

                yield from self._generate_mini_batches(
                    cache_id_list,
                    cache_chunks,
                    shuffle,
                    state,
                )

    def _generate_mini_batches(
        self,
        id_list: List[str],
        batches: Dict[str, List[torch.Tensor]],
        shuffle: bool,
        state: np.random.RandomState,
    ):
        if shuffle:
            indices = np.arange(0, len(id_list))
            state.shuffle(indices)
            batches = {k: [v[i] for i in indices] for k, v in batches.items()}
            id_list = [id_list[i] for i in indices]

        bs = self.batch_size
        while len(id_list) >= bs:
            # Make mini-batch and yield
            yield (
                id_list[:bs],
                {k: torch.stack(v[:bs], 0) for k, v in batches.items()},
            )
            id_list = id_list[bs:]
            batches = {k: v[bs:] for k, v in batches.items()}

        return id_list, batches
