<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.11-blue"></a>
</p>

## Installation

### Install Conda (Optional):

#### Linux
```sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n funasr python=3.7
conda activate funasr
```
#### Mac
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
# For M1 chip
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX*
source ~/.zashrc
conda create -n funasr python=3.8
conda activate funasr
```
#### Windows
Ref to [docs](https://docs.conda.io/en/latest/miniconda.html#windows-installers)

### Install Pytorch (version >= 1.11.0):

```sh
pip3 install torch torchaudio
```
If there exists CUDAs in your environments, you should install the pytorch with the version matching the CUDA. The matching list could be found in [docs](https://pytorch.org/get-started/previous-versions/).
### Install funasr

#### Install from pip

```shell
pip3 install -U funasr
# For the users in China, you could install with the command:
# pip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

#### Or install from source code

``` sh
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip3 install -e ./
# For the users in China, you could install with the command:
# pip3 install -e ./ -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

### Install modelscope (Optional)
If you want to use the pretrained models in ModelScope, you should install the modelscope:

```shell
pip3 install -U modelscope
# For the users in China, you could install with the command:
# pip3 install -U modelscope -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

### FQA
- For installation on MAC M1 chip, the following error may happen:
- - _cffi_backend.cpython-38-darwin.so' (mach-o file, but is an incompatible architecture (have (x86_64), need (arm64e)))
    ```shell
    pip uninstall cffi pycparser
    ARCHFLAGS="-arch arm64" pip install cffi pycparser --compile --no-cache-dir
    ```
