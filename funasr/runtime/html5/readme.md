# online asr demo for html5

## requirement
### python
```shell
flask
gevent
pyOpenSSL
```

### javascript
```shell
Recorder [html5录音](https://github.com/xiangyuecn/Recorder)
```
## html5服务配置
### 启动html5服务

```shell
usage: h5Server.py [-h] [--host HOST] [--port PORT] [--certfile CERTFILE]
                   [--keyfile KEYFILE]
python h5Server.py --port 1337
```
注:
wsconnecter.js里配置online asr wss路径
var Uri = "wss://xxx:xxx/" 

### 浏览器打开地址
https://127.0.0.1:1337/static/index.html


### demo页面如下
![img](https://github.com/alibaba-damo-academy/FunASR/blob/for-html5-demo/funasr/runtime/html5/demo.gif)


##后端配置
h5打开麦克风需要https协议，同时后端的asr websocket也必须是wss协议，而目前[online asr](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket)模型只支持ws协议，所以我们通过nginx配置实现wss协议到ws协议的转换。

##具体过程如下：
浏览器htts --> html5 demo服务 --> js wss接口 --> nginx服务 --> ws asr online srv

##配置nginx wss协议(了解的可以跳过）
生成证书(注意这种证书并不能被所有浏览器认可，部分手动授权可以访问,最好使用其他认证的官方ssl证书)

### 生成私钥，按照提示填写内容
openssl genrsa -des3 -out server.key 1024
 
### 生成csr文件 ，按照提示填写内容
openssl req -new -key server.key -out server.csr
 
### 去掉pass
cp server.key server.key.org 
openssl rsa -in server.key.org -out server.key
 
### 生成crt文件，有效期1年（365天）
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt

##nginx转发配置示例
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


      proxy_pass http://127.0.0.1:1111/;  # asr online model ws address:port
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