from modelscope.hub.snapshot_download import snapshot_download
import sys

if __name__ == "__main__":
    model_tag = sys.argv[1]
    local_model_dir = sys.argv[2]
    model_dir = snapshot_download(model_tag, cache_dir=local_model_dir, revision='1.0.0')