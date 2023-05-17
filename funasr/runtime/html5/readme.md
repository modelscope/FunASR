# online asr demo for html5

## requirement
### python
```shell
flask
gevent
pyOpenSSL
```

### javascript
[html5 recorder.js](https://github.com/xiangyuecn/Recorder)
```shell
Recorder 
```

### demo
![img](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/html5/demo.gif)

## wss or ws protocol for ws_server_online
1) wss: browser microphone data --> html5 demo server --> js wss api --> wss asr online srv #for certificate generation just look back

2) ws: browser microphone data  --> html5 demo server --> js wss api --> nginx wss server --> ws asr online srv

## 1.html5 demo start
### ssl certificate is required

```shell
usage: h5Server.py [-h] [--host HOST] [--port PORT] [--certfile CERTFILE]
                   [--keyfile KEYFILE]
python h5Server.py --port 1337
```
## 2.asr online srv start
[detail for online asr](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket)
Online asr provides wss or ws way. if started in ws way, nginx is required for relay.
### wss way, ssl certificate is required
```shell
python ws_server_online.py --certfile server.crt --keyfile server.key  --port 5921
```
### ws way
```shell
python ws_server_online.py  --port 5921
```
## 3.modify asr address in wsconnecter.js according to your environment
asr address in wsconnecter.js must be wss, just like
var Uri = "wss://xxx:xxx/" 

## 4.open browser to access html5 demo
https://youraddress:port/static/index.html




## certificate generation by yourself
generated certificate may not suitable for all browsers due to security concerns. you'd better buy or download an authenticated ssl certificate from authorized agency.

```shell
### 1) Generate a private key
openssl genrsa -des3 -out server.key 1024
 
### 2) Generate a csr file
openssl req -new -key server.key -out server.csr
 
### 3) Remove pass
cp server.key server.key.org 
openssl rsa -in server.key.org -out server.key
 
### 4) Generated a crt file, valid for 1 year
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt
```

## nginx configuration (you can skip it if you known)
https and wss protocol are required by browsers when want to open microphone and websocket.  
if [online asr](https://github.com/alibaba-damo-academy/FunASR/tree/main/funasr/runtime/python/websocket) run in ws way, you should use nginx to convert wss to ws.
### nginx wss->ws configuration example
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
2. We acknowledge [AiHealthx](http://www.aihealthx.com/) for contributing the html5 demo.