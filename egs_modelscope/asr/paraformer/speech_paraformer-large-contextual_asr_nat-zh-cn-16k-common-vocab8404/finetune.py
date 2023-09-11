import os

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer

from funasr.datasets.ms_dataset import MsDataset
from funasr.utils.modelscope_param import modelscope_args


def modelscope_finetune(params):
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir, exist_ok=True)
    # dataset split ["train", "validation"]
    ds_dict = MsDataset.load(params.data_path)
    kwargs = dict(
        model=params.model,
        model_revision="v1.0.2",
        data_dir=ds_dict,
        dataset_type=params.dataset_type,
        work_dir=params.output_dir,
        batch_bins=params.batch_bins,
        max_epoch=params.max_epoch,
        lr=params.lr)
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    params = modelscope_args(model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404", data_path="./data")
    params.output_dir = "./checkpoint"              # 模型保存路径
    params.data_path = "./example_data/"            # 数据路径
    params.dataset_type = "large"                   # finetune contextual_paraformer模型只能使用large dataset
    params.batch_bins = 120000                      # batch size，如果dataset_type="small"，batch_bins单位为fbank特征帧数，如果dataset_type="large"，batch_bins单位为毫秒，
    params.max_epoch = 20                           # 最大训练轮数
    params.lr = 0.0002                              # 设置学习率

    modelscope_finetune(params)