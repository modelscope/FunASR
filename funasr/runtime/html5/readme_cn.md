# Html5 server for asr service

## Requirement
### python
```shell
pip install flask
# pip install gevent (Optional)
# pip install pyOpenSSL (Optional)
```

### javascript (Optional)
[html5录音](https://github.com/xiangyuecn/Recorder)
```shell
Recorder 
```

## demo页面如下
<div align="center"><img src="./demo.gif" width="150"/> </div>

## 两种ws_server连接模式
### 1)直接连接模式，浏览器https麦克风 --> html5 demo服务 --> js wss接口 --> wss asr online srv(证书生成请往后看)

### 2)nginx中转，浏览器https麦克风 --> html5 demo服务 --> js wss接口 --> nginx服务 --> ws asr online srv

## 操作步骤
### html5 demo服务启动
#### 启动html5服务，需要ssl证书(已生成，如需要自己生成请往后看)
用法如下:
```shell
h5Server.py [-h] [--host HOST] [--port PORT] [--certfile CERTFILE] [--keyfile KEYFILE]             
```
例子如下，需要注意ip地址，如果从其他设备访问需求（例如手机端），需要将ip地址设为真实ip 
```shell
python h5Server.py --host 0.0.0.0 --port 1337 --keyfile server.key
# python h5Server.py --host 30.220.136.139 --port 1337 --keyfile server.key
```
### 启动ws or wss asr online srv
[具体请看online asr](../python/websocket)
online asr提供两种wss和ws模式，wss模式可以直接启动，无需nginx中转。否则需要通过nginx将wss转发到该online asr的ws端口上

`Tips:` asr 服务需要与html5服务部署到同一个物理机器上
#### wss方式
```shell
cd ../python/websocket
python ws_server_online.py --certfile ../../html5/server.crt --keyfile ../../html5/server.key  --port 1095
```
#### ws方式
```shell
cd ../python/websocket
python ws_server_online.py  --port 1095
```

### 浏览器打开地址
ip地址需要与html5 server保持一致，如果是本地机器，可以用127.0.0.1
```shell
https://127.0.0.1:1337/static/index.html
# https://30.220.136.139:1337/static/index.html
```

### 修改网页里asr接口地址
修改网页中，asr服务器地址（websocket srv的ip与端口），点击开始即可使用。注意h5服务和asr服务需要在同一个服务器上，否则存在跨域问题。


## 自行生成证书
生成证书(注意这种证书并不能被所有浏览器认可，部分手动授权可以访问,最好使用其他认证的官方ssl证书)

```shell
### 1)生成私钥，按照提示填写内容
openssl genrsa -des3 -out server.key 1024
 
### 2)生成csr文件 ，按照提示填写内容
openssl req -new -key server.key -out server.csr
 
### 去掉pass
cp server.key server.key.org 
openssl rsa -in server.key.org -out server.key
 
### 生成crt文件，有效期1年（365天）
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt
```

## nginx配置说明(了解的可以跳过)
h5打开麦克风需要https协议，同时后端的asr websocket也必须是wss协议，如果[online asr](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket)以ws方式运行，我们可以通过nginx配置实现wss协议到ws协议的转换。

### nginx转发配置示例
```shell
events {                                                                                                            [0/1548]
    worker_connections  1024;
    accept_mutex on;
  }
http {
  error_log  error.log;
  access_log  access.log;
  server {

    listen 5921 ssl http2;  # nginx listen port for wss
    server_name www.test.com;

    ssl_certificate     /funasr/server.crt;
    ssl_certificate_key /funasr/server.key;
    ssl_protocols       TLSv1 TLSv1.1 TLSv1.2;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    location /wss/ {


      proxy_pass http://127.0.0.1:1111/;  # asr online model ws address and port
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
      proxy_read_timeout 600s;

    }
  }
```
### 修改wsconnecter.js里asr接口地址
wsconnecter.js里配置online asr服务地址路径，这里配置的是wss端口
var Uri = "wss://xxx:xxx/wss/" 
## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [AiHealthx](http://www.aihealthx.com/) for contributing the html5 demo.