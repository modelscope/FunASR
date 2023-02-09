# 安装
FunASR的安装十分便捷，下面将给出详细的安装步骤：

- 安装Conda并创建虚拟环境
``` sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
conda create -n funasr python=3.7
conda activate funasr
```

- 安装Pytorch (版本 >= 1.7.0):

```sh
pip install torch torchvision torchaudio
```

关于更多的版本, 请参照 [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

- 安装 ModelScope

对于国内用户，可以通过配置下述镜像源来加快下载速度
```sh
pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple
```

安装或更新ModelScope
``` sh
pip install "modelscope[audio]" --upgrade -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

- 下载FunASR仓库，并安装剩余所需依赖
``` sh
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip install --editable ./
```