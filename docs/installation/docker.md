# Docker

## Install Docker

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
Ref to [docs](https://docs.docker.com/desktop/install/windows-install/)

## Start Docker
```shell
sudo systemctl start docker
```
## Download image

### Image
#### CPU
`registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-py37-torch1.11.0-tf1.15.5-1.5.0`

#### GPU

`registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-1.5.0`

### Pull Image
```shell
sudo docker pull <image-name>:<tag>
```

### Check Image 
```shell
sudo docker images
```

## Run Docker
```shell
sudo docker run -itd --name funasr <image-name>:<tag> bash
sudo docker exec -it funasr bash
```

## Stop Docker
```shell
exit
sudo docker ps
sudo docker stop funasr
```

