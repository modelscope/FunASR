import numpy as np
from torch.utils.data import DataLoader

from funasr.datasets.iterable_dataset import IterableESPnetDataset
from funasr.datasets.small_datasets.collate_fn import CommonCollateFn
from funasr.datasets.small_datasets.preprocessor import build_preprocess


def build_streaming_iterator(
        task_name,
        preprocess_args,
        data_path_and_name_and_type,
        key_file: str = None,
        batch_size: int = 1,
        fs: dict = None,
        mc: bool = False,
        dtype: str = np.float32,
        num_workers: int = 1,
        use_collate_fn: bool = True,
        preprocess_fn=None,
        ngpu: int = 0,
        train: bool = False,
) -> DataLoader:
    """Build DataLoader using iterable dataset"""

    # preprocess
    if preprocess_fn is not None:
        preprocess_fn = preprocess_fn
    elif preprocess_args is not None:
        preprocess_args.task_name = task_name
        preprocess_fn = build_preprocess(preprocess_args, train)
    else:
        preprocess_fn = None

    # collate
    if not use_collate_fn:
        collate_fn = None
    elif task_name in ["punc", "lm"]:
        collate_fn = CommonCollateFn(int_pad_value=0)
    else:
        collate_fn = CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)
    if collate_fn is not None:
        kwargs = dict(collate_fn=collate_fn)
    else:
        kwargs = {}

    dataset = IterableESPnetDataset(
        data_path_and_name_and_type,
        float_dtype=dtype,
        fs=fs,
        mc=mc,
        preprocess=preprocess_fn,
        key_file=key_file,
    )
    if dataset.apply_utt2category:
        kwargs.update(batch_size=1)
    else:
        kwargs.update(batch_size=batch_size)

    return DataLoader(
        dataset=dataset,
        pin_memory=ngpu > 0,
        num_workers=num_workers,
        **kwargs,
    )
