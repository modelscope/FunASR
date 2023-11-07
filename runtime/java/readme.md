# Client for java websocket example

 

## Building for Linux/Unix

### install java environment
```shell
#  in ubuntu
apt-get install openjdk-11-jdk
```

 

### Build and run by make


```shell
cd funasr/runtime/java
# download java lib
make downjar
# compile 
make buildwebsocket
# run client
make runclient

```

## Run java websocket client by shell

```shell
# full command refer to Makefile runclient
usage:  FunasrWsClient [-h] [--port PORT] [--host HOST] [--audio_in AUDIO_IN] [--num_threads NUM_THREADS]
                 [--chunk_size CHUNK_SIZE] [--chunk_interval CHUNK_INTERVAL] [--mode MODE]

Where:
   --host <string>
     (required)  server-ip

   --port <int>
     (required)  port

   --audio_in <string>
     (required)  the wav or pcm file path

   --num_threads <int>
     thread number for test
   
   --mode
     asr mode, support "offline" "online" "2pass"

 

example:
FunasrWsClient --host localhost --port 8889 --audio_in ./asr_example.wav --num_threads 1 --mode 2pass

result json, example like:
{"mode":"offline","text":"欢迎大家来体验达摩院推出的语音识别模型","wav_name":"javatest"}
```


## Acknowledge
1. This project is maintained by [FunASR community](https://github.com/alibaba-damo-academy/FunASR).
2. We acknowledge [zhaoming](https://github.com/zhaomingwork/FunASR/tree/java-ws-client-support/funasr/runtime/java) for contributing the java websocket client example.


