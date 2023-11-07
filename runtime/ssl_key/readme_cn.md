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
