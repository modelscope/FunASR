# Installation
FunASR is easy to install, which is mainly based on python packages.

- Clone the repo
``` sh
git clone https://github.com/alibaba/FunASR.git
```

- Install Conda
``` sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
conda create -n funasr python=3.7
conda activate funasr
```

- Install Pytorch (version >= 1.7.0):

| cuda  | |
|:-----:| --- |
|  9.2  | conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch |
| 10.2  | conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch |
| 11.1  | conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch |

For more versions, please see [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

- Install ModelScope
``` sh
pip install "modelscope[audio_asr]" --upgrade -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

- Install other packages
``` sh
pip install --editable ./
```