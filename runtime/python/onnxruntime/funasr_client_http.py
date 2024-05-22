import base64
import requests
import threading


with open("A2_0.wav", "rb") as f:
    test_wav_bytes = f.read()
url = "http://127.0.0.1:8888/api/asr"


def send_post(i, url, wav_bytes):
    r1 = requests.post(url, json={"wav_base64": str(base64.b64encode(wav_bytes), "utf-8")})
    print("线程:", i, r1.json())


for i in range(100):
    t = threading.Thread(
        target=send_post,
        args=(
            i,
            url,
            test_wav_bytes,
        ),
    )
    t.start()
    # t.join()
print("完成测试")
