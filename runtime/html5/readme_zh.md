(简体中文|[English](./readme.md))

# 语音识别服务Html5客户端访问界面

服务端部署采用websocket协议，客户端可以支持html5网页访问，支持麦克风输入与文件输入，可以通过如下2种方式访问：
- 方式一：

   html客户端直连，手动下载客户端（[点击此处](https://github.com/modelscope/FunASR/tree/main/runtime/html5/static)）至本地，打开`index.html`网页，输入wss地址与端口号

- 方式二：

   html5服务端，自动下载客户端至本地，支持手机等端上访问

## 语音识别服务启动

支持python版本与c++版本服务部署，其中

- python版本
  
  直接部署python pipeline，支持流式实时语音识别模型，离线语音识别模型，流式离线一体化纠错模型，输出带标点文字。单个server，支持单个client。

- c++版本
  
  funasr-runtime-sdk，支持一键部署，0.1.0版本，支持离线文件转写。单个server，支持上百路client请求。

### python版本服务启动

#### 安装依赖环境

```shell
pip3 install -U modelscope funasr flask
# 中国大陆用户，如果遇到网络问题，可以通过下面指令安装：
# pip3 install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
git clone https://github.com/alibaba/FunASR.git && cd FunASR
```

#### 启动ASR服务

#### wss方式

```shell
cd funasr/runtime/python/websocket
python funasr_wss_server.py --port 10095
```

详细参数配置与解析（[点击此处](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket)）

#### html5服务（可选）

如果需要使用上面所说的客户端方式二，进行访问，可以启动html5服务
```shell
h5Server.py [-h] [--host HOST] [--port PORT] [--certfile CERTFILE] [--keyfile KEYFILE]             
```
例子如下，需要注意ip地址，如果从其他设备访问需求（例如手机端），需要将ip地址设为真实公网ip 
```shell
cd funasr/runtime/html5
python h5Server.py --host 0.0.0.0 --port 1337
```

启动后，在浏览器中输入（[https://127.0.0.1:1337/static/index.html](https://127.0.0.1:1337/static/index.html)）即可访问

### c++ 版本服务启动

由于c++依赖环境较多，建议采用docker部署，支持一键启动服务

```shell
curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy-offline-cpu-zh.sh;
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh install --workspace /root/funasr-runtime-resources
```
详细参数配置与解析（[点击此处](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/docs/SDK_tutorial_zh.md)）


## 客户端测试

### 方式一

html客户端直连，手动下载客户端（[点击此处](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/html5/static)）至本地，打开`index.html`网页，输入wss地址与端口号即可使用

### 方式二

html5服务端，自动下载客户端至本地，支持手机等端上访问，ip地址需要与html5 server保持一致，如果是本地机器，可以用127.0.0.1


```shell
https://127.0.0.1:1337/static/index.html
```

输入wss地址与端口号即可使用


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [AiHealthx](http://www.aihealthx.com/) for contributing the html5 demo.
