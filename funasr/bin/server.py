"""
FunASR Server — OpenAI-compatible speech recognition API.

Usage:
    funasr-server                          # default: sensevoice on cuda:0, port 8000
    funasr-server --device cpu --port 9000
    funasr-server --model paraformer
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="FunASR OpenAI-Compatible API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  funasr-server                          # Start with SenseVoice on GPU
  funasr-server --device cpu             # Start on CPU
  funasr-server --model paraformer       # Use Paraformer model
  funasr-server --port 9000             # Custom port

Then use with OpenAI SDK:
  from openai import OpenAI
  client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
  result = client.audio.transcriptions.create(model="sensevoice", file=open("a.wav","rb"))
""",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--device", default="cuda", help="Device: cuda, cpu, mps (default: cuda)")
    parser.add_argument("--model", default="auto", help="Pre-load model: auto (GPU=fun-asr-nano, CPU=sensevoice), sensevoice, paraformer, fun-asr-nano")
    args = parser.parse_args()

    try:
        import uvicorn
        import fastapi
    except ImportError:
        print("Error: funasr-server requires additional packages.")
        print("Install with: pip install fastapi uvicorn python-multipart")
        sys.exit(1)

    # Import and configure the app
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'examples', 'openai_api'))

    # Use inline app to avoid path issues
    from funasr.bin._server_app import create_app

    app = create_app(device=args.device, preload_model=args.model)

    print(f"╔══════════════════════════════════════════════╗")
    print(f"║  FunASR Server v1.3.2                        ║")
    print(f"║  Device: {args.device:<8}                          ║")
    print(f"║  Model:  {args.model:<12}                      ║")
    print(f"║  URL:    http://{args.host}:{args.port}/v1          ║")
    print(f"║  Docs:   http://{args.host}:{args.port}/docs        ║")
    print(f"╚══════════════════════════════════════════════╝")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
