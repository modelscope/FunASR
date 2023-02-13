
class modelscope_args():
	def __init__(self,
	            task: str = "",
	            model: str = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
	            data_path: str = None,
	            output_dir: str = None,
	            model_revision: str = "master",
	            dataset_type: str = "small",
	            batch_bins: int = 2000,
	            max_epoch: int = None,
	            lr: float = None,
	            ):
		self.task = task
		self.model = model
		self.data_path = data_path
		self.output_dir = output_dir
		self.model_revision = model_revision
		self.dataset_type = dataset_type
		self.batch_bins = batch_bins
		self.max_epoch = max_epoch
		self.lr = lr
		
		
		