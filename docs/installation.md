<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.11-blue"></a>
</p>

## Installation

### Install Conda (Optional):

```sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n funasr python=3.7
conda activate funasr
```

### Install Pytorch (version >= 1.11.0):

```sh
pip install torch torchaudio
```

For more details about torch, please see [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

### Install funasr

#### Install from pip

```shell
pip install -U funasr
# For the users in China, you could install with the command:
# pip install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

#### Or install from source code

``` sh
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install -e ./
# For the users in China, you could install with the command:
# pip install -e ./ -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

### Install modelscope (Optional)
If you want to use the pretrained models in ModelScope, you should install the modelscope:

```shell
pip install -U modelscope
# For the users in China, you could install with the command:
# pip install -U modelscope -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html -i https://mirror.sjtu.edu.cn/pypi/web/simple
```