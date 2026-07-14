import json
import subprocess
import sys
import unittest
from pathlib import Path


SERVER = Path(__file__).with_name("funasr_mcp.py")


class FunASRMCPTransportTest(unittest.TestCase):
    def test_stdio_uses_newline_delimited_json_rpc(self):
        requests = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "transport-test", "version": "1.0"},
                },
            },
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ]
        stdin = "".join(f"{json.dumps(request)}\n" for request in requests)

        completed = subprocess.run(
            [sys.executable, str(SERVER)],
            input=stdin,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual([response["id"] for response in responses], [1, 2])
        self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "funasr")
        self.assertEqual(
            responses[1]["result"]["tools"][0]["name"], "transcribe_audio"
        )


if __name__ == "__main__":
    unittest.main()
