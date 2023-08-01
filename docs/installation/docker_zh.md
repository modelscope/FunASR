(简体中文|[English](./docker.md))

# Docker

## 安装Docker

### Ubuntu
```shell
curl -fsSL https://test.docker.com -o test-docker.sh
sudo sh test-docker.sh
```
### Debian
```shell
 curl -fsSL https://get.docker.com -o get-docker.sh
 sudo sh get-docker.sh
```

### CentOS
```shell
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
```

### MacOS
```shell
brew install --cask --appdir=/Applications docker
```

### Windows
请参考[文档](https://docs.docker.com/desktop/install/windows-install/)

## 启动Docker
```shell
sudo systemctl start docker
```
## 下载Docker镜像

### 镜像仓库

#### CPU
`registry.cn-hangzhou.aliyuncs.com/funasr_repo/funasr:funasr-runtime-sdk-cpu-0.1.0`

#### GPU

`registry.cn-beijing.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.7.0`

### 拉取镜像
```shell
sudo docker pull <image-name>:<tag>
```

### 查看镜像
```shell
sudo docker images
```

## 运行Docker
```shell
# cpu
sudo docker run -itd --name funasr -v <local_dir:dir_in_docker> <image-name>:<tag> /bin/bash
# gpu
sudo docker run -itd --gpus all --name funasr -v <local_dir:dir_in_docker> <image-name>:<tag> /bin/bash

sudo docker exec -it funasr /bin/bash
```

## 停止Docker
```shell
exit
sudo docker ps
sudo docker stop funasr
```

