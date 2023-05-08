# Service with websocket-python

This is a demo using funasr pipeline with websocket python-api. 

## For the Server

### Install the modelscope and funasr

```shell
pip install -U modelscope funasr
# For the users in China, you could install with the command:
# pip install -U modelscope funasr -i https://mirror.sjtu.edu.cn/pypi/web/simple
git clone https://github.com/alibaba/FunASR.git && cd FunASR
```

### Install the requirements for server

```shell
cd funasr/runtime/python/websocket
pip install -r requirements_server.txt
```

### Start server
#### ASR offline server
```shell
python ws_server_offline.py --port 10095 --asr_model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
```

#### ASR streaming server
```shell
python ws_server_online.py --port 10095 --asr_model_online "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
```

#### ASR offline/online 2pass server

```shell
python ws_server_2pass.py --port 10095 --asr_model "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"  --asr_model_online "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
```

## For the client

Install the requirements for client
```shell
git clone https://github.com/alibaba/FunASR.git && cd FunASR
cd funasr/runtime/python/websocket
pip install -r requirements_client.txt
```

### Start client

```shell
python ws_client.py \
--host [ip_address] \
--port [port id] \
--chunk_size ["5,10,5"=600ms, "8,8,4"=480ms] \
--chunk_interval [duration of send chunk_size/chunk_interval] \
--words_max_print [max number of words to print] \
--audio_in [if set, loadding from wav.scp, else recording from mircrophone] \
--output_dir [if set, write the results to output_dir] \
--send_without_sleep [only set for offline]
```

#### ASR offline client
##### Recording from mircrophone
```shell
# --chunk_interval, "10": 600/10=60ms, "5"=600/5=120ms, "20": 600/12=30ms
python ws_client.py --host "0.0.0.0" --port 10095 --chunk_interval 10 --words_max_print 100
```
##### Loadding from wav.scp(kaldi style)
```shell
# --chunk_interval, "10": 600/10=60ms, "5"=600/5=120ms, "20": 600/12=30ms
python ws_client.py --host "0.0.0.0" --port 10095 --chunk_interval 10 --words_max_print 100 --audio_in "./data/wav.scp" --send_without_sleep --output_dir "./results"
```

#### ASR streaming client
##### Recording from mircrophone
```shell
# --chunk_size, "5,10,5"=600ms, "8,8,4"=480ms
python ws_client.py --host "0.0.0.0" --port 10095 --chunk_size "5,10,5" --words_max_print 100
```
##### Loadding from wav.scp(kaldi style)
```shell
# --chunk_size, "5,10,5"=600ms, "8,8,4"=480ms
python ws_client.py --host "0.0.0.0" --port 10095 --chunk_size "5,10,5" --audio_in "./data/wav.scp" --words_max_print 100 --output_dir "./results"
```

#### ASR offline/online 2pass client
##### Recording from mircrophone
```shell
# --chunk_size, "5,10,5"=600ms, "8,8,4"=480ms
python ws_client.py --host "0.0.0.0" --port 10095 --chunk_size "8,8,4" --words_max_print 10000
```
##### Loadding from wav.scp(kaldi style)
```shell
# --chunk_size, "5,10,5"=600ms, "8,8,4"=480ms
python ws_client.py --host "0.0.0.0" --port 10095 --chunk_size "8,8,4" --audio_in "./data/wav.scp" --words_max_print 10000 --output_dir "./results"
```
## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [zhaoming](https://github.com/zhaomingwork/FunASR/tree/fix_bug_for_python_websocket) for contributing the websocket service.
3. We acknowledge [cgisky1980](https://github.com/cgisky1980/FunASR) for contributing the websocket service of offline model.
