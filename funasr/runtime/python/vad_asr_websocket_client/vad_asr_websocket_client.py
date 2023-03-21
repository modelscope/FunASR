#""" from https://github.com/cgisky1980/550W_AI_Assistant """

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
import logging
logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)
import websocket
import pyaudio
import time
import json
import threading


# ---------WebsocketClient相关  主要处理 on_message on_open  已经做了断线重连处理
class WebsocketClient(object):
    def __init__(self, address, message_callback=None):
        super(WebsocketClient, self).__init__()
        self.address = address
        self.message_callback = None

    def on_message(self, ws, message):
        try:
            messages = json.loads(
                (message.encode("raw_unicode_escape")).decode()
            )  # 收到WS消息后的处理
            if messages.get("type") == "ping":
                self.ws.send('{"type":"pong"}')
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
        except KeyError:
            print("KeyError!")

    def on_error(self, ws, error):
        print("client error:", error)

    def on_close(self, ws):
        print("### client closed ###")
        self.ws.close()
        self.is_running = False

    def on_open(self, ws):  # 连上ws后发布登录信息
        self.is_running = True
        self.ws.send(
            '{"type":"login","uid":"asr","pwd":"tts9102093109"}'
        )  # WS链接上后的登陆处理

    def close_connect(self):
        self.ws.close()

    def send_message(self, message):
        try:
            self.ws.send(message)
        except BaseException as err:
            pass

    def run(self):  # WS初始化
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            self.address,
            on_message=lambda ws, message: self.on_message(ws, message),
            on_error=lambda ws, error: self.on_error(ws, error),
            on_close=lambda ws: self.on_close(ws),
        )
        websocket.enableTrace(False)  # 要看ws调试信息，请把这行注释掉
        self.ws.on_open = lambda ws: self.on_open(ws)
        self.is_running = False
        # WS断线重连判断
        while True:  
            if not self.is_running:
                self.ws.run_forever()
            time.sleep(3)  # 3秒检测一次


class WSClient(object):
    def __init__(self, address, call_back):
        super(WSClient, self).__init__()
        self.client = WebsocketClient(address, call_back)
        self.client_thread = None

    def run(self):
        self.client_thread = threading.Thread(target=self.run_client)
        self.client_thread.start()

    def run_client(self):
        self.client.run()

    def send_message(self, message):
        self.client.send_message(message)


def vad(data):  # VAD推理
    segments_result = vad_pipline(audio_in=data)
    if segments_result["text"] == "[]":
        return False
    else:
        return True


# 创建一个VAD对象
vad_pipline = pipeline(
    task=Tasks.voice_activity_detection,
    model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    model_revision="v1.2.0",
    output_dir=None,
    batch_size=1,
)

param_dict = dict()
param_dict["hotword"] = "小五 小五月"  # 设置热词，用空格隔开


# 创建一个ASR对象
inference_pipeline2 = pipeline(
    task=Tasks.auto_speech_recognition,
    model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    param_dict=param_dict,
)

# 创建一个PyAudio对象
p = pyaudio.PyAudio()

# 定义一些参数
FORMAT = pyaudio.paInt16  # 采样格式
CHANNELS = 1  # 单声道
RATE = 16000  # 采样率
CHUNK = int(RATE / 1000 * 300)  # 每个片段的帧数（300毫秒）
RECORD_NUM = 0 # 录制时长（片段）

# 打开输入流
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

print("开始...")

# 创建一个WS连接
ws_client = WSClient("ws://localhost:7272", None)
ws_client.run()

frames = []  # 存储所有的帧数据
buffer = []  # 存储缓存中的帧数据（最多两个片段）
silence_count = 0  # 统计连续静音的次数
speech_detected = False  # 标记是否检测到语音

# 循环读取输入流中的数据
while True:
    data = stream.read(CHUNK)  # 读取一个片段的数据
    buffer.append(data)  # 将当前数据添加到缓存中

    if len(buffer) > 2:
        buffer.pop(0)  # 如果缓存超过两个片段，则删除最早的一个

    if speech_detected:
        frames.append(data)
        RECORD_NUM += 1
        # print(str(RECORD_NUM)+ "\r")

    if vad(data):  # VAD 判断是否有声音
        if not speech_detected:
            print("开始录音...")
            speech_detected = True  # 标记为检测到语音
            frames = []
            frames.extend(buffer)  # 把之前2个语音数据快加入
        silence_count = 0  # 重置静音次数

    else:
        silence_count += 1  # 增加静音次数
        #检测静音次数4次  或者已经录了50个数据块，则录音停止
        if speech_detected and (silence_count > 4 or RECORD_NUM > 50):  
            print("停止录音...")
            audio_in = b"".join(frames)
            rec_result = inference_pipeline2(audio_in=audio_in)  # ws播报数据
            rec_result["type"] = "nlp"  # 添加ws播报数据
            ws_client.send_message(
                json.dumps(rec_result, ensure_ascii=False)
            )  # ws发送到服务端
            print(rec_result)
            frames = []  # 清空所有的帧数据
            buffer = []  # 清空缓存中的帧数据（最多两个片段）
            silence_count = 0  # 统计连续静音的次数清零
            speech_detected = False  # 标记是否检测到语音
            RECORD_NUM = 0

print("结束录制...")

# 停止并关闭输入流
stream.stop_stream()
stream.close()

# 关闭PyAudio对象
p.terminate()
