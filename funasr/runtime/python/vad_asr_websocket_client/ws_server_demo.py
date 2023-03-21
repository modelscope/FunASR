# server.py
import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        print(message) # 打印收到的消息
        await websocket.send(message)

start_server = websockets.serve(echo, "localhost", 7272)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()