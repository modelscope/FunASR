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
