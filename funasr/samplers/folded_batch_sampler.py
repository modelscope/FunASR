from typing import Iterator
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union

from typeguard import check_argument_types

from funasr.fileio.read_text import load_num_sequence_text
from funasr.fileio.read_text import read_2column_text
from funasr.samplers.abs_sampler import AbsSampler


class FoldedBatchSampler(AbsSampler):
    def __init__(
        self,
        batch_size: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        fold_lengths: Sequence[int],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        utt2category_file: str = None,
    ):
        assert check_argument_types()
        assert batch_size > 0
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_size = batch_size
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")

        category2utt = {}
        if utt2category_file is not None:
            utt2category = read_2column_text(utt2category_file)
            if set(utt2category) != set(first_utt2shape):
                raise RuntimeError(
                    "keys are mismatched between "
                    f"{utt2category_file} != {shape_files[0]}"
                )
            for k in keys:
                category2utt.setdefault(utt2category[k], []).append(k)
        else:
            category2utt["default_category"] = keys

        self.batch_list = []
        for d, v in category2utt.items():
            category_keys = v
            # Decide batch-sizes
            start = 0
            batch_sizes = []
            while True:
                k = category_keys[start]
                factor = max(int(d[k][0] / m) for d, m in zip(utt2shapes, fold_lengths))
                bs = max(min_batch_size, int(batch_size / (1 + factor)))
                if self.drop_last and start + bs > len(category_keys):
                    # This if-block avoids 0-batches
                    if len(self.batch_list) > 0:
                        break

                bs = min(len(category_keys) - start, bs)
                batch_sizes.append(bs)
                start += bs
                if start >= len(category_keys):
                    break

            if len(batch_sizes) == 0:
                # Maybe we can't reach here
                raise RuntimeError("0 batches")

            # If the last batch-size is smaller than minimum batch_size,
            # the samples are redistributed to the other mini-batches
            if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
                for i in range(batch_sizes.pop(-1)):
                    batch_sizes[-(i % len(batch_sizes)) - 2] += 1

            if not self.drop_last:
                # Bug check
                assert sum(batch_sizes) == len(
                    category_keys
                ), f"{sum(batch_sizes)} != {len(category_keys)}"

            # Set mini-batch
            cur_batch_list = []
            start = 0
            for bs in batch_sizes:
                assert len(category_keys) >= start + bs, "Bug"
                minibatch_keys = category_keys[start : start + bs]
                start += bs
                if sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif sort_in_batch == "ascending":
                    # Key are already sorted in ascending
                    pass
                else:
                    raise ValueError(
                        "sort_in_batch must be ascending or "
                        f"descending: {sort_in_batch}"
                    )
                cur_batch_list.append(tuple(minibatch_keys))

            if sort_batch == "ascending":
                pass
            elif sort_batch == "descending":
                cur_batch_list.reverse()
            else:
                raise ValueError(
                    f"sort_batch must be ascending or descending: {sort_batch}"
                )
            self.batch_list.extend(cur_batch_list)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_size={self.batch_size}, "
            f"shape_files={self.shape_files}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
