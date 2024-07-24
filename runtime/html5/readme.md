([简体中文](./readme_zh.md)|English)

# Speech Recognition Service Html5 Client Access Interface

The server deployment uses the websocket protocol. The client can support html5 webpage access and microphone input or file input. There are two ways to access the service:
- Method 1: 

  Directly connect to the html client, manually download the client ([click here](https://github.com/modelscope/FunASR/tree/main/runtime/html5/static)) to the local computer, and open the index.html webpage to enter the wss address and port number.

- Method 2: 

   Html5 server, automatically download the client to the local computer, and support access by mobile phones and other devices.

## Starting Speech Recognition Service

Support the deployment of Python and C++ versions, where

- Python version
  
  Directly deploy the Python pipeline, support streaming real-time speech recognition models, offline speech recognition models, streaming offline integrated error correction models, and output text with punctuation marks. Single server, supporting a single client.

- C++ version
  
  funasr-runtime-sdk, supports one-key deployment, version 0.1.0, supports offline file transcription. Single server, supporting requests from hundreds of clients.

### Starting Python Version Service

#### Install Dependencies

```shell
pip3 install -U modelscope funasr flask
# Users in mainland China, if encountering network issues, can install with the following command:
# pip3 install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
git clone https://github.com/alibaba/FunASR.git && cd FunASR
```

#### Start ASR Service

#### wss Method

```shell
cd funasr/runtime/python/websocket
python funasr_wss_server.py --port 10095
```

For detailed parameter configuration and analysis, please click [here](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket).

#### Html5 Service (Optional)

If you need to use the client method mentioned above to access it, you can start the html5 service

```shell
h5Server.py [-h] [--host HOST] [--port PORT] [--certfile CERTFILE] [--keyfile KEYFILE]             
```
As shown in the example below, pay attention to the IP address. If accessing from another device (such as a mobile phone), you need to set the IP address to the real public IP address.
```shell
cd funasr/runtime/html5
python h5Server.py --host 0.0.0.0 --port 1337
```

After starting, enter ([https://127.0.0.1:1337/static/index.html](https://127.0.0.1:1337/static/index.html)) in the browser to access it.

### Starting C++ Version Service

Since there are many dependencies for C++, it is recommended to deploy it using docker, which supports one-key start of the service.


```shell
curl -O https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/shell/funasr-runtime-deploy-offline-cpu-zh.sh;
sudo bash funasr-runtime-deploy-offline-cpu-zh.sh install --workspace /root/funasr-runtime-resources
```
For detailed parameter configuration and analysis, please click [here](https://github.com/alibaba-damo-academy/FunASR/blob/main/funasr/runtime/docs/SDK_tutorial_zh.md).

## Client Testing

### Method 1

Directly connect to the html client, manually download the client ([click here](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/html5/static)) to the local computer, and open the index.html webpage, enter the wss address and port number to use.

### Method 2

Html5 server, automatically download the client to the local computer, and support access by mobile phones and other devices. The IP address needs to be consistent with the html5 server. If it is a local computer, you can use 127.0.0.1.

```shell
https://127.0.0.1:1337/static/index.html
```

Enter the wss address and port number to use.


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [AiHealthx](http://www.aihealthx.com/) for contributing the html5 demo.
