# online asr demo for html5

## requirement
### python
```shell
flask
gevent
pyOpenSSL
```

### javascript
[html5录音](https://github.com/xiangyuecn/Recorder)
```shell
Recorder 
```

### demo页面如下
![img](https://github.com/alibaba-damo-academy/FunASR/blob/for-html5-demo/funasr/runtime/html5/demo.gif)

## 两种ws_server_online连接模式
### 1)直接连接模式，浏览器https麦克风 --> html5 demo服务 --> js wss接口 --> wss asr online srv(证书生成请往后看)
```shell
python ws_server_online.py --certfile server.crt --keyfile server.key  --port 5921
```
### 2)nginx中转，浏览器https麦克风 --> html5 demo服务 --> js wss接口 --> nginx服务 --> ws asr online srv
```shell
python ws_server_online.py  --port 5921
```
## 1.html5 demo服务启动
### 启动html5服务，需要ssl证书(自己生成请往后看)

```shell
usage: h5Server.py [-h] [--host HOST] [--port PORT] [--certfile CERTFILE]
                   [--keyfile KEYFILE]
python h5Server.py --port 1337
```
## 2.启动ws or wss asr online srv
[具体请看online asr](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket)
online asr只提供两种ws和wss模式，wss模式可以直接启动，无需nginx中转。否则需要通过nginx将wss转发到该online asr的ws端口上

## 3.修改wsconnecter.js里asr接口地址
wsconnecter.js里配置online asr服务地址路径，这里配置的是wss端口
var Uri = "wss://xxx:xxx/" 

## 4.浏览器打开地址测试
https://127.0.0.1:1337/static/index.html





## nginx配置说明(了解的可以跳过)
h5打开麦克风需要https协议，同时后端的asr websocket也必须是wss协议，而目前[online asr](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket)模型只支持ws协议，所以我们通过nginx配置实现wss协议到ws协议的转换。

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

## nginx转发配置示例
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

## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [爱医声](http://www.aihealthx.com/) for contributing the html5 demo.