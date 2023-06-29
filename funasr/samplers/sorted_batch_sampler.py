import logging
from typing import Iterator
from typing import Tuple


from funasr.fileio.read_text import load_num_sequence_text
from funasr.samplers.abs_sampler import AbsSampler


class SortedBatchSampler(AbsSampler):
    """BatchSampler with sorted samples by length.

    Args:
        batch_size:
        shape_file:
        sort_in_batch: 'descending', 'ascending' or None.
        sort_batch:
    """

    def __init__(
        self,
        batch_size: int,
        shape_file: str,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
    ):
        assert batch_size > 0
        self.batch_size = batch_size
        self.shape_file = shape_file
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shape = load_num_sequence_text(shape_file, loader_type="csv_int")
        if sort_in_batch == "descending":
            # Sort samples in descending order (required by RNN)
            keys = sorted(utt2shape, key=lambda k: -utt2shape[k][0])
        elif sort_in_batch == "ascending":
            # Sort samples in ascending order
            keys = sorted(utt2shape, key=lambda k: utt2shape[k][0])
        else:
            raise ValueError(
                f"sort_in_batch must be either one of "
                f"ascending, descending, or None: {sort_in_batch}"
            )
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_file}")

        # Apply max(, 1) to avoid 0-batches
        N = max(len(keys) // batch_size, 1)
        if not self.drop_last:
            # Split keys evenly as possible as. Note that If N != 1,
            # the these batches always have size of batch_size at minimum.
            self.batch_list = [
                keys[i * len(keys) // N : (i + 1) * len(keys) // N] for i in range(N)
            ]
        else:
            self.batch_list = [
                tuple(keys[i * batch_size : (i + 1) * batch_size]) for i in range(N)
            ]

        if len(self.batch_list) == 0:
            logging.warning(f"{shape_file} is empty")

        if sort_in_batch != sort_batch:
            if sort_batch not in ("ascending", "descending"):
                raise ValueError(
                    f"sort_batch must be ascending or descending: {sort_batch}"
                )
            self.batch_list.reverse()

        if len(self.batch_list) == 0:
            raise RuntimeError("0 batches")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_file={self.shape_file}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
