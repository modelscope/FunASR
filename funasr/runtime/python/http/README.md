# Service with http-python

## Server

1. Install requirements

```shell
cd funasr/runtime/python/http
pip install -r requirements.txt
```

2. Start server

```shell
python server.py --port 8000
```

More parameters:
```shell
python server.py \
--host [host ip] \
--port [server port] \
--asr_model [asr model_name] \
--punc_model [punc model_name] \
--ngpu [0 or 1] \
--ncpu [1 or 4] \
--certfile [path of certfile for ssl] \
--keyfile [path of keyfile for ssl] \
--temp_dir [upload file temp dir] 
```

## Client

```shell
# get test audio file
wget https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav
python client.py --host=127.0.0.1 --port=8000 --audio_path=asr_example_zh.wav
```

More parameters:
```shell
python server.py \
--host [sever ip] \
--port [sever port] \
--add_pun [add pun to result] \
--audio_path [use audio path] 
```


## 支持多进程

方法是启动多个`server.py`，然后通过Nginx的负载均衡分发请求，达到支持多用户同时连效果，处理方式如下，默认您已经安装了Nginx，没安装的请参考[官方安装教程](https://nginx.org/en/linux_packages.html#Ubuntu)。

配置Nginx。
```shell
sudo cp -f asr_nginx.conf /etc/nginx/nginx.conf
sudo service nginx reload
```

然后使用脚本启动多个服务，每个服务的端口号不一样。
```shell
sudo chmod +x start_server.sh
./start_server.sh
```

**说明：** 默认是3个进程，如果需要修改，首先修改`start_server.sh`的最后那部分，可以添加启动数量。然后修改`asr_nginx.conf`配置文件的`upstream backend`部分，增加新启动的服务，可以使其他服务器的服务。
