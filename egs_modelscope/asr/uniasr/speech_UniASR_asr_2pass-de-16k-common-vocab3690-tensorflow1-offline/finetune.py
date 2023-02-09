import os
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from funasr.datasets.ms_dataset import MsDataset


def modelscope_finetune(params):
    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"], exist_ok=True)
    # dataset split ["train", "validation"]
    ds_dict = MsDataset.load(params["data_dir"])
    kwargs = dict(
        model=params["model"],
        model_revision=params["model_revision"],
        data_dir=ds_dict,
        dataset_type=params["dataset_type"],
        work_dir=params["output_dir"],
        batch_bins=params["batch_bins"],
        max_epoch=params["max_epoch"],
        lr=params["lr"])
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    params = {}
    params["output_dir"] = "./checkpoint"
    params["data_dir"] = "./data"
    params["batch_bins"] = 2000
    params["dataset_type"] = "small"
    params["max_epoch"] = 50
    params["lr"] = 0.00005
    params["model"] = "damo/speech_UniASR_asr_2pass-de-16k-common-vocab3690-tensorflow1-offline"
    params["model_revision"] = None
    modelscope_finetune(params)
