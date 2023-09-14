import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="127.0.0.1",
                    required=False,
                    help="sever ip")
parser.add_argument("--port",
                    type=int,
                    default=8000,
                    required=False,
                    help="server port")
parser.add_argument("--add_pun",
                    type=int,
                    default=1,
                    required=False,
                    help="add pun to result")
parser.add_argument("--audio_path",
                    type=str,
                    default='asr_example_zh.wav',
                    required=False,
                    help="use audio path")
args = parser.parse_args()
print("-----------  Configuration Arguments -----------")
for arg, value in vars(args).items():
    print("%s: %s" % (arg, value))
print("------------------------------------------------")


url = f'http://{args.host}:{args.port}/recognition'
data = {'add_pun': args.add_pun}
headers = {}
files = [('audio', ('file', open(args.audio_path, 'rb'), 'application/octet-stream'))]

response = requests.post(url, headers=headers, data=data, files=files)
print(response.text)
