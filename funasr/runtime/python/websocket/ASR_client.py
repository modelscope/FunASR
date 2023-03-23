import pyaudio
# import websocket #区别服务端这里是 websocket-client库
import time
import websockets
import asyncio
from queue import Queue
# import threading
voices = Queue()
async def hello():
    global ws # 定义一个全局变量ws，用于保存websocket连接对象
    uri = "ws://localhost:8899"
    ws = await websockets.connect(uri, subprotocols=["binary"]) # 创建一个长连接
    ws.max_size = 1024 * 1024 * 20
    print("connected ws server")
async def send(data):
    global ws # 引用全局变量ws
    try:
        await ws.send(data) # 通过ws对象发送数据
    except Exception as e:
        print('Exception occurred:', e)
    


asyncio.get_event_loop().run_until_complete(hello()) # 启动协程  


# 其他函数可以通过调用send(data)来发送数据，例如：
async def test():
    #print("2")
    global voices
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = int(RATE / 1000 * 300)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    while True:

        data = stream.read(CHUNK)
        
        voices.put(data)
        #print(voices.qsize())
        await asyncio.sleep(0.01)
    
      



async def ws_send():
    global voices
    print("started to sending data!")
    while True:
        while not voices.empty():
            data = voices.get()
            voices.task_done()
            await send(data)
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.01)

async def main():
    task = asyncio.create_task(test()) # 创建一个后台任务
    task2 = asyncio.create_task(ws_send()) # 创建一个后台任务
     
    await asyncio.gather(task, task2)

asyncio.run(main())