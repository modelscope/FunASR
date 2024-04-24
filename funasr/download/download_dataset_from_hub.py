def download_dataset():
    pass


def download_dataset_from_ms(**kwargs):
    from modelscope.msdatasets import MsDataset

    dataset_name = kwargs.get("dataset_name", "speech_asr/speech_asr_aishell1_trainsets")
    subset_name = kwargs.get("subset_name", "default")
    split = kwargs.get("split", "train")
    data_dump_dir = kwargs.get("data_dump_dir", None)
    ds = MsDataset.load(
        dataset_name=dataset_name, subset_name=subset_name, split=split, cache_dir=data_dump_dir
    )
