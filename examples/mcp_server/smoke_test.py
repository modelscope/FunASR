#!/usr/bin/env python3
import argparse
import json
import subprocess


REQUESTS = [
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "container-smoke-test", "version": "1.0"},
        },
    },
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
]


def validate_responses(responses):
    by_id = {response.get("id"): response for response in responses}
    if set(by_id) != {1, 2}:
        raise ValueError(f"expected response IDs 1 and 2, got {sorted(by_id)}")

    server_info = by_id[1].get("result", {}).get("serverInfo", {})
    if server_info.get("name") != "funasr":
        raise ValueError(f"unexpected serverInfo: {server_info}")

    tools = by_id[2].get("result", {}).get("tools", [])
    tool_names = {tool.get("name") for tool in tools}
    if "transcribe_audio" not in tool_names:
        raise ValueError(f"transcribe_audio missing from tools/list: {tool_names}")


def run_smoke_test(image, timeout):
    payload = "".join(f"{json.dumps(request)}\n" for request in REQUESTS)
    completed = subprocess.run(
        ["docker", "run", "--rm", "-i", image],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"container exited with {completed.returncode}: {completed.stderr.strip()}"
        )

    responses = [
        json.loads(line) for line in completed.stdout.splitlines() if line.strip()
    ]
    validate_responses(responses)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the FunASR MCP image")
    parser.add_argument("image", help="Local or remote container image reference")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    run_smoke_test(args.image, args.timeout)
    print(f"MCP container smoke test passed: {args.image}")


if __name__ == "__main__":
    main()
