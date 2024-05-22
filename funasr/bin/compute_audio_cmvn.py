import os
import json
import numpy as np
import torch
import hydra
import logging
from omegaconf import DictConfig, OmegaConf

from funasr.register import tables
from funasr.download.download_from_hub import download_model
from funasr.train_utils.set_all_random_seed import set_all_random_seed


@hydra.main(config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
    if kwargs.get("debug", False):
        import pdb

        pdb.set_trace()

    assert "model" in kwargs
    if "model_conf" not in kwargs:
        logging.info("download models from model hub: {}".format(kwargs.get("hub", "ms")))
        kwargs = download_model(is_training=kwargs.get("is_training", True), **kwargs)

    main(**kwargs)


def main(**kwargs):
    print(kwargs)
    # set random seed
    # tables.print()
    set_all_random_seed(kwargs.get("seed", 0))
    torch.backends.cudnn.enabled = kwargs.get("cudnn_enabled", torch.backends.cudnn.enabled)
    torch.backends.cudnn.benchmark = kwargs.get("cudnn_benchmark", torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic = kwargs.get("cudnn_deterministic", True)

    tokenizer = kwargs.get("tokenizer", None)

    # build frontend if frontend is none None
    frontend = kwargs.get("frontend", None)
    if frontend is not None:
        frontend_class = tables.frontend_classes.get(frontend)
        frontend = frontend_class(**kwargs["frontend_conf"])
        kwargs["frontend"] = frontend
        kwargs["input_size"] = frontend.output_size()

    # dataset
    dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset"))
    dataset_train = dataset_class(
        kwargs.get("train_data_set_list"),
        frontend=frontend,
        tokenizer=None,
        is_training=False,
        **kwargs.get("dataset_conf")
    )

    # dataloader
    batch_sampler = kwargs["dataset_conf"].get("batch_sampler", "BatchSampler")
    batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
    dataset_conf = kwargs.get("dataset_conf")
    dataset_conf["batch_type"] = "example"
    dataset_conf["batch_size"] = 1
    dataset_conf["num_workers"] = os.cpu_count() or 32
    batch_sampler_train = batch_sampler_class(dataset_train, is_training=False, **dataset_conf)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, collate_fn=dataset_train.collator, **batch_sampler_train
    )

    iter_stop = int(kwargs.get("scale", 1.0) * len(dataloader_train))

    total_frames = 0
    for batch_idx, batch in enumerate(dataloader_train):
        if batch_idx >= iter_stop:
            break

        fbank = batch["speech"].numpy()[0, :, :]
        if total_frames == 0:
            mean_stats = np.sum(fbank, axis=0)
            var_stats = np.sum(np.square(fbank), axis=0)
        else:
            mean_stats += np.sum(fbank, axis=0)
            var_stats += np.sum(np.square(fbank), axis=0)
        total_frames += fbank.shape[0]

    cmvn_info = {
        "mean_stats": list(mean_stats.tolist()),
        "var_stats": list(var_stats.tolist()),
        "total_frames": total_frames,
    }
    cmvn_file = kwargs.get("cmvn_file", "cmvn.json")
    # import pdb;pdb.set_trace()
    with open(cmvn_file, "w") as fout:
        fout.write(json.dumps(cmvn_info))

    mean = -1.0 * mean_stats / total_frames
    var = 1.0 / np.sqrt(var_stats / total_frames - mean * mean)
    dims = mean.shape[0]
    am_mvn = os.path.dirname(cmvn_file) + "/am.mvn"
    with open(am_mvn, "w") as fout:
        fout.write(
            "<Nnet>"
            + "\n"
            + "<Splice> "
            + str(dims)
            + " "
            + str(dims)
            + "\n"
            + "[ 0 ]"
            + "\n"
            + "<AddShift> "
            + str(dims)
            + " "
            + str(dims)
            + "\n"
        )
        mean_str = str(list(mean)).replace(",", "").replace("[", "[ ").replace("]", " ]")
        fout.write("<LearnRateCoef> 0 " + mean_str + "\n")
        fout.write("<Rescale> " + str(dims) + " " + str(dims) + "\n")
        var_str = str(list(var)).replace(",", "").replace("[", "[ ").replace("]", " ]")
        fout.write("<LearnRateCoef> 0 " + var_str + "\n")
        fout.write("</Nnet>" + "\n")


"""
python funasr/bin/compute_audio_cmvn.py \
--config-path "/Users/zhifu/funasr1.0/examples/aishell/paraformer/conf" \
--config-name "train_asr_paraformer_conformer_12e_6d_2048_256.yaml" \
++train_data_set_list="/Users/zhifu/funasr1.0/data/list/audio_datasets.jsonl" \
++cmvn_file="/Users/zhifu/funasr1.0/data/list/cmvn.json" \
++dataset_conf.num_workers=0
"""
if __name__ == "__main__":
    main_hydra()
