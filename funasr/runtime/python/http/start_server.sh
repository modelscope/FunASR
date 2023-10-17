#!/bin/bash

# 创建日志文件夹
if [ ! -d "log/" ];then
  mkdir log
fi

# kill掉之前的进程
server_id=`ps -ef | grep server.py | grep -v "grep" | awk '{print $2}'`
echo $server_id

for id in $server_id
do
    kill -9 $id
    echo "killed $id"
done

# 启动多个服务，可以设置使用不同的显卡
CUDA_VISIBLE_DEVICES=0 nohup python -u server.py --host=localhost --port=8001 >> log/output1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u server.py --host=localhost --port=8002 >> log/output2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u server.py --host=localhost --port=8003 >> log/output3.log 2>&1 &
