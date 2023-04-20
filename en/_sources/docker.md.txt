# Docker

## Install 

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

`registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-1.5.0'

### Pull Image
```shell
sudo docker pull <image-name>:<tag>
```

### Check Downloaded Image 
```shell
sudo docker images
```

### Run Docker
```shell
sudo docker run -it <image-name>:<tag> bash
```

### Stop Docker
```shell
sudo docker ps
sudo docker stop <container-id>
```

