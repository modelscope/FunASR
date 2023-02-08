class modelscope_args():
    def __init__(self,
                 task: str = "",
                 model: str = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                 data_path: str = None,
                 output_dir: str = None,
                 model_revision: str = None,
                 dataset_type: str = "small",
                 batch_bins: int = 2000,
                 max_epoch: int = None,
                 accum_grad: int = None,
                 keep_nbest_models: int = None,
                 optim: str = None,
                 lr: float = None,
                 scheduler: str = None,
                 scheduler_conf: dict = None,
                 specaug: str = None,
                 specaug_conf: dict = None,
                 ):
        self.task = task
        self.model = model
        self.data_path = data_path
        self.output_dir = output_dir
        self.model_revision = model_revision
        self.dataset_type = dataset_type
        self.batch_bins = batch_bins
        self.max_epoch = max_epoch
        self.accum_grad = accum_grad
        self.keep_nbest_models = keep_nbest_models
        self.optim = optim
        self.lr = lr
        self.scheduler = scheduler
        self.scheduler_conf = scheduler_conf
        self.specaug = specaug
        self.specaug_conf = specaug_conf
