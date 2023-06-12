# Html5 server for asr service

## Requirement
#### Install the modelscope and funasr
```shell
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
git clone https://github.com/alibaba/FunASR.git && cd FunASR
```
#### Install the requirements for server
```shell
pip install flask
# pip install gevent (Optional)
# pip install pyOpenSSL (Optional)
```

### javascript (Optional)
[html5 recorder.js](https://github.com/xiangyuecn/Recorder)
```shell
Recorder 
```

## demo
<div align="center"><img src="./demo.gif" width="150"/> </div>

## Steps
### Html5 demo

```shell
usage: h5Server.py [-h] [--host HOST] [--port PORT] [--certfile CERTFILE] [--keyfile KEYFILE]
```
`e.g.`
```shell
cd funasr/runtime/html5
python h5Server.py --host 0.0.0.0 --port 1337 
```
### asr service
[detail for asr](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket)

`Tips:` asr service and html5 service should be deployed on the same device.
```shell
cd ../python/websocket
python wss_srv_asr.py --port 10095
```


### open browser to access html5 demo
```shell
https://127.0.0.1:1337/static/index.html
# https://30.220.136.139:1337/static/index.html
```

### modify asr address in html according to your environment
asr address in index.html must be wss


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [AiHealthx](http://www.aihealthx.com/) for contributing the html5 demo.