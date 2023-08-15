(简体中文|[English](./installation.md))

<p align="left">
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-brightgreen.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python->=3.7,<=3.10-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Pytorch-%3E%3D1.11-blue"></a>
</p>

## 安装

### 安装Conda（可选）：

#### Linux
```sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n funasr python=3.8
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

### 安装Pytorch（版本 >= 1.11.0）：

```sh
pip3 install torch torchaudio
```
如果您的环境中存在CUDAs，则应安装与CUDA匹配版本的pytorch，匹配列表可在文档中找到（[文档](https://pytorch.org/get-started/previous-versions/)）。
### 安装funasr

#### 从pip安装

```shell
pip3 install -U funasr
# 对于中国大陆用户，可以使用以下命令进行安装：
# pip3 install -U funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

#### 或者从源代码安装

``` sh
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip3 install -e ./
# 对于中国大陆用户，可以使用以下命令进行安装：
# pip3 install -e ./ -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

### 安装modelscope（可选）

如果您想要使用ModelScope中的预训练模型，则应安装modelscope:

```shell
pip3 install -U modelscope
# 对于中国大陆用户，可以使用以下命令进行安装：
# pip3 install -U modelscope -i https://mirror.sjtu.edu.cn/pypi/web/simple
```

### 常见问题解答
- 在MAC M1芯片上安装时，可能会出现以下错误：
- - _cffi_backend.cpython-38-darwin.so' (mach-o file, but is an incompatible architecture (have (x86_64), need (arm64e)))
    ```shell
    pip uninstall cffi pycparser
    ARCHFLAGS="-arch arm64" pip install cffi pycparser --compile --no-cache-dir
    ```
