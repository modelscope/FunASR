## certificate generation by yourself
generated certificate may not suitable for all browsers due to security concerns. you'd better buy or download an authenticated ssl certificate from authorized agency.

```shell
### 1) Generate a private key
openssl genrsa -des3 -out server.key 2048
 
### 2) Generate a csr file
openssl req -new -key server.key -out server.csr
 
### 3) Remove pass
cp server.key server.key.org 
openssl rsa -in server.key.org -out server.key
 
### 4) Generated a crt file, valid for 1 year
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt
```
