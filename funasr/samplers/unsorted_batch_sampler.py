import logging
from typing import Iterator
from typing import Tuple


from funasr.fileio.read_text import read_2column_text
from funasr.samplers.abs_sampler import AbsSampler


class UnsortedBatchSampler(AbsSampler):
    """BatchSampler with constant batch-size.

    Any sorting is not done in this class,
    so no length information is required,
    This class is convenient for decoding mode,
    or not seq2seq learning e.g. classification.

    Args:
        batch_size:
        key_file:
    """

    def __init__(
        self,
        batch_size: int,
        key_file: str,
        drop_last: bool = False,
        utt2category_file: str = None,
    ):
        assert batch_size > 0
        self.batch_size = batch_size
        self.key_file = key_file
        self.drop_last = drop_last

        # utt2shape:
        #    uttA <anything is o.k>
        #    uttB <anything is o.k>
        utt2any = read_2column_text(key_file)
        if len(utt2any) == 0:
            logging.warning(f"{key_file} is empty")
        # In this case the, the first column in only used
        keys = list(utt2any)
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {key_file}")

        category2utt = {}
        if utt2category_file is not None:
            utt2category = read_2column_text(utt2category_file)
            if set(utt2category) != set(keys):
                raise RuntimeError(
                    f"keys are mismatched between {utt2category_file} != {key_file}"
                )
            for k, v in utt2category.items():
                category2utt.setdefault(v, []).append(k)
        else:
            category2utt["default_category"] = keys

        self.batch_list = []
        for d, v in category2utt.items():
            category_keys = v
            # Apply max(, 1) to avoid 0-batches
            N = max(len(category_keys) // batch_size, 1)
            if not self.drop_last:
                # Split keys evenly as possible as. Note that If N != 1,
                # the these batches always have size of batch_size at minimum.
                cur_batch_list = [
                    category_keys[i * len(keys) // N : (i + 1) * len(keys) // N]
                    for i in range(N)
                ]
            else:
                cur_batch_list = [
                    tuple(category_keys[i * batch_size : (i + 1) * batch_size])
                    for i in range(N)
                ]
            self.batch_list.extend(cur_batch_list)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"key_file={self.key_file}, "
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
