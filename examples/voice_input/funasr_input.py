"""
FunASR Voice Input — 语音输入法

安装:
    pip install funasr sounddevice numpy pyperclip openai

使用:
    1. 先启动 funasr-server: funasr-server --device cuda (或 cpu)
    2. 运行: python funasr_input.py
    3. 按住 空格键 说话，松开自动识别并输入到光标位置

配置:
    --server   FunASR server 地址 (默认 http://localhost:8000/v1)
    --key      快捷键 (默认 ctrl+shift+space)
    --lang     语言 (默认 auto)
"""

import argparse
import sys
import os
import tempfile
import threading
import time
import io
import wave
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="FunASR Voice Input - 语音输入法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用方法:
  1. 启动 FunASR 服务: funasr-server --device cuda
  2. 运行本程序: python funasr_input.py
  3. 按 Ctrl+Shift+Space 开始录音，再按一次停止并输入文字

示例:
  python funasr_input.py                          # 默认配置
  python funasr_input.py --server http://gpu:8000/v1  # 远程服务器
  python funasr_input.py --model paraformer       # 指定模型
""",
    )
    parser.add_argument("--server", default="http://localhost:8000/v1", help="FunASR server URL")
    parser.add_argument("--model", default="sensevoice", help="ASR model")
    parser.add_argument("--lang", default="auto", help="Language hint")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--hotkey", default="ctrl+shift+space", help="Hotkey to toggle recording")
    args = parser.parse_args()

    try:
        import sounddevice as sd
    except ImportError:
        print("Error: sounddevice required. Install: pip install sounddevice")
        sys.exit(1)
    try:
        import pyperclip
    except ImportError:
        print("Error: pyperclip required. Install: pip install pyperclip")
        sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai required. Install: pip install openai")
        sys.exit(1)
    try:
        from pynput import keyboard
    except ImportError:
        print("Error: pynput required. Install: pip install pynput")
        sys.exit(1)

    client = OpenAI(base_url=args.server, api_key="not-needed")

    # State
    recording = False
    audio_frames = []
    lock = threading.Lock()

    def start_recording():
        nonlocal recording, audio_frames
        with lock:
            if recording:
                return
            recording = True
            audio_frames = []
        print("🎤 录音中... (再按快捷键停止)")

        def callback(indata, frames, time_info, status):
            if recording:
                audio_frames.append(indata.copy())

        sd.default.samplerate = args.rate
        sd.default.channels = 1
        sd.default.dtype = 'int16'

        stream = sd.InputStream(callback=callback)
        stream.start()

        # Wait until recording stops
        while recording:
            time.sleep(0.05)
        stream.stop()
        stream.close()

    def stop_and_transcribe():
        nonlocal recording
        with lock:
            if not recording:
                return
            recording = False

        print("⏳ 识别中...")

        if not audio_frames:
            print("❌ 没有录到音频")
            return

        # Convert to WAV
        audio_data = np.concatenate(audio_frames, axis=0)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(args.rate)
            wf.writeframes(audio_data.tobytes())
        wav_buffer.seek(0)

        # Send to FunASR server
        try:
            result = client.audio.transcriptions.create(
                model=args.model,
                file=("recording.wav", wav_buffer, "audio/wav"),
            )
            text = result.text.strip()
            if text:
                # Copy to clipboard and simulate paste
                pyperclip.copy(text)
                print(f"✅ {text}")
                print("   (已复制到剪贴板，Cmd+V/Ctrl+V 粘贴)")

                # Try to auto-paste (macOS)
                if sys.platform == 'darwin':
                    os.system('osascript -e \'tell application "System Events" to keystroke "v" using command down\'')
                elif sys.platform == 'linux':
                    os.system('xdotool key ctrl+v 2>/dev/null')
            else:
                print("❌ 未识别到语音")
        except Exception as e:
            print(f"❌ 识别失败: {e}")
            print(f"   请确认 funasr-server 正在运行: funasr-server --device cuda")

    # Hotkey handling
    print("=" * 50)
    print("  FunASR 语音输入法")
    print("=" * 50)
    print(f"  服务器: {args.server}")
    print(f"  模型:   {args.model}")
    print(f"  快捷键: {args.hotkey}")
    print(f"  语言:   {args.lang}")
    print("=" * 50)
    print(f"按 {args.hotkey} 开始/停止录音")
    print("按 Ctrl+C 退出")
    print()

    # Parse hotkey
    hotkey_parts = args.hotkey.lower().split('+')
    modifiers = set()
    key_char = None
    for part in hotkey_parts:
        if part in ('ctrl', 'control'):
            modifiers.add(keyboard.Key.ctrl_l)
        elif part in ('shift',):
            modifiers.add(keyboard.Key.shift_l)
        elif part in ('alt', 'option'):
            modifiers.add(keyboard.Key.alt_l)
        elif part in ('cmd', 'command', 'super'):
            modifiers.add(keyboard.Key.cmd)
        elif part == 'space':
            key_char = keyboard.Key.space
        else:
            key_char = keyboard.KeyCode.from_char(part)

    current_modifiers = set()
    record_thread = None

    def on_press(key):
        nonlocal record_thread
        if key in modifiers or (hasattr(key, 'value') and key.value in [k.value for k in modifiers if hasattr(k, 'value')]):
            current_modifiers.add(key)
        if key == key_char and len(current_modifiers) >= len(modifiers):
            if not recording:
                record_thread = threading.Thread(target=start_recording, daemon=True)
                record_thread.start()
            else:
                stop_and_transcribe()

    def on_release(key):
        current_modifiers.discard(key)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("\n👋 退出")


if __name__ == "__main__":
    main()
