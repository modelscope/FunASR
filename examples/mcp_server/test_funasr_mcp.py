import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SERVER = Path(__file__).with_name("funasr_mcp.py")


def load_server_module():
    spec = importlib.util.spec_from_file_location("funasr_mcp_test_module", SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        source_version = (SERVER.parents[2] / "funasr" / "version.txt").read_text().strip()
        self.assertEqual(
            responses[0]["result"]["serverInfo"]["version"], source_version
        )
        self.assertEqual(responses[1]["result"]["tools"][0]["name"], "transcribe_audio")


class FunASRMCPConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.server = load_server_module()

    def test_get_model_honors_model_and_device_environment(self):
        model = object()
        auto_model = Mock(return_value=model)
        fake_funasr = types.SimpleNamespace(AutoModel=auto_model)

        with patch.dict(sys.modules, {"funasr": fake_funasr}), patch.dict(
            os.environ,
            {"FUNASR_MODEL": "custom/model", "FUNASR_DEVICE": "cuda:1"},
        ):
            self.assertIs(self.server.get_model(), model)

        auto_model.assert_called_once_with(
            model="custom/model",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:1",
            disable_update=True,
        )

    def test_initialize_reports_installed_funasr_version(self):
        with tempfile.TemporaryDirectory() as package_dir:
            (Path(package_dir) / "version.txt").write_text("9.9.9")
            package_spec = types.SimpleNamespace(
                submodule_search_locations=[package_dir]
            )
            with patch.object(
                self.server, "find_spec", return_value=package_spec
            ), patch.object(self.server, "send_response") as send_response:
                self.server.handle_request({"id": 1, "method": "initialize"})

        server_info = send_response.call_args[0][1]["serverInfo"]
        self.assertEqual(server_info["version"], "9.9.9")

    def test_transcribe_forwards_language_to_model(self):
        model = Mock()
        model.generate.return_value = [{"text": "<|yue|><|Speech|> nei hou"}]

        with patch.object(self.server, "get_model", return_value=model):
            result = self.server.transcribe("audio.wav", language="yue")

        model.generate.assert_called_once_with(
            input="audio.wav", batch_size=1, language="yue"
        )
        self.assertEqual(result, {"text": "nei hou"})


class FunASRMCPToolContractTest(unittest.TestCase):
    def setUp(self):
        self.server = load_server_module()

    def test_tool_contract_matches_default_sensevoice_capabilities(self):
        with patch.object(self.server, "send_response") as send_response:
            self.server.handle_request({"id": 1, "method": "tools/list"})

        tool = send_response.call_args[0][1]["tools"][0]
        description = tool["description"]
        language_schema = tool["inputSchema"]["properties"]["language"]

        self.assertIn("Mandarin", description)
        self.assertIn("Cantonese", description)
        self.assertNotIn("50+", description)
        self.assertNotIn("diarization", description.lower())
        self.assertNotIn("timestamp", description.lower())
        self.assertEqual(
            language_schema["enum"], ["auto", "zh", "yue", "en", "ja", "ko"]
        )

    def test_empty_audio_path_returns_a_parameter_error(self):
        with patch.object(self.server, "send_response") as send_response:
            self.server.handle_request(
                {
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "transcribe_audio",
                        "arguments": {"audio_path": ""},
                    },
                }
            )

        result = send_response.call_args[0][1]
        self.assertTrue(result["isError"])
        self.assertIn("audio_path is required", result["content"][0]["text"])

    def test_unsupported_language_is_rejected_before_inference(self):
        with tempfile.NamedTemporaryFile() as audio_file, patch.object(
            self.server, "transcribe"
        ) as transcribe, patch.object(
            self.server, "send_response"
        ) as send_response:
            self.server.handle_request(
                {
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "transcribe_audio",
                        "arguments": {
                            "audio_path": audio_file.name,
                            "language": "fr",
                        },
                    },
                }
            )

        transcribe.assert_not_called()
        result = send_response.call_args[0][1]
        self.assertTrue(result["isError"])
        self.assertIn("unsupported language", result["content"][0]["text"])

    def test_user_home_is_expanded_before_file_validation(self):
        with patch.object(
            self.server.os.path,
            "expanduser",
            return_value="/home/user/audio.wav",
        ) as expanduser, patch.object(
            self.server.os.path, "isfile", return_value=True
        ), patch.object(
            self.server, "transcribe", return_value={"text": "hello"}
        ) as transcribe, patch.object(
            self.server, "send_response"
        ):
            self.server.handle_request(
                {
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "transcribe_audio",
                        "arguments": {"audio_path": "~/audio.wav"},
                    },
                }
            )

        expanduser.assert_called_once_with("~/audio.wav")
        transcribe.assert_called_once_with("/home/user/audio.wav", "auto")

    def test_inference_error_is_returned_without_crashing_server(self):
        with tempfile.NamedTemporaryFile() as audio_file, patch.object(
            self.server, "transcribe", side_effect=RuntimeError("model unavailable")
        ), patch.object(self.server, "send_response") as send_response:
            self.server.handle_request(
                {
                    "id": 5,
                    "method": "tools/call",
                    "params": {
                        "name": "transcribe_audio",
                        "arguments": {"audio_path": audio_file.name},
                    },
                }
            )

        result = send_response.call_args[0][1]
        self.assertTrue(result["isError"])
        self.assertIn("model unavailable", result["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
