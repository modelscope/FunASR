import os
import shutil

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.hub.snapshot_download import snapshot_download


if __name__ == '__main__':
    import sys
    
    model = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    checkpoint_name = sys.argv[3]
    
    try:
        pretrained_model_path = snapshot_download(model, cache_dir=checkpoint_dir)
    except BaseException:
        raise BaseException(f"Please download pretrain model from ModelScope firstly.")
    shutil.copy(os.path.join(checkpoint_dir, checkpoint_name), os.path.join(pretrained_model_path, "model.pb"))
