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
        data_dir=ds_dict,
        dataset_type=params.dataset_type,
        work_dir=params.output_dir,
        batch_bins=params.batch_bins,
        max_epoch=params.max_epoch,
        lr=params.lr)
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    params = modelscope_args(model="damo/speech_UniASR_asr_2pass-fa-16k-common-vocab1257-pytorch-online", data_path="./data")
    params.output_dir = "./checkpoint"              # m模型保存路径
    params.data_path = "./example_data/"            # 数据路径
    params.dataset_type = "small"                   # 小数据量设置small，若数据量大于1000小时，请使用large
    params.batch_bins = 2000                       # batch size，如果dataset_type="small"，batch_bins单位为fbank特征帧数，如果dataset_type="large"，batch_bins单位为毫秒，
    params.max_epoch = 20                           # 最大训练轮数
    params.lr = 0.00005                             # 设置学习率
    
    modelscope_finetune(params)
