# Installation
FunASR is easy to install. The detailed installation steps are as follows:

- Install Conda and create virtual environment:
```sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n funasr python=3.7
conda activate funasr
```

- Install Pytorch (version >= 1.7.0):
```sh
pip install torch torchaudio
```

For more versions, please see [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

- Install ModelScope

For users in China, you can configure the following mirror source to speed up the downloading:
``` sh
pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple
```
Install or update ModelScope
```sh
pip install "modelscope[audio_asr]" --upgrade -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

- Clone the repo and install other packages
``` sh
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install --editable ./
```