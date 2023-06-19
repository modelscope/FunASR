from modelscope.hub.snapshot_download import snapshot_download
import sys


cache_dir = sys.argv[1]
model_dir = snapshot_download('damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch', cache_dir=cache_dir)
