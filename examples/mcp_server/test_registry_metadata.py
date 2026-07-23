import importlib.util
import json
import re
import unittest
from pathlib import Path


MCP_DIR = Path(__file__).resolve().parent
REPO_ROOT = MCP_DIR.parents[1]
SERVER_JSON = MCP_DIR / "server.json"
DOCKERFILE = MCP_DIR / "Dockerfile"
README = MCP_DIR / "README.md"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-mcp-server.yml"
SMOKE_TEST = MCP_DIR / "smoke_test.py"


def load_smoke_test_module():
    spec = importlib.util.spec_from_file_location("funasr_mcp_smoke_test", SMOKE_TEST)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load smoke test module from {SMOKE_TEST}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MCPRegistryMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads(SERVER_JSON.read_text())

    def test_server_metadata_uses_canonical_namespace(self):
        self.assertEqual(
            self.metadata["$schema"],
            "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
        )
        self.assertEqual(self.metadata["name"], "io.github.modelscope/funasr-mcp")
        self.assertLessEqual(len(self.metadata["description"]), 100)
        self.assertEqual(
            self.metadata["repository"],
            {
                "url": "https://github.com/modelscope/FunASR",
                "source": "github",
                "id": "569959091",
                "subfolder": "examples/mcp_server",
            },
        )

    def test_oci_package_version_matches_server_version(self):
        packages = self.metadata["packages"]
        self.assertEqual(len(packages), 1)
        package = packages[0]
        self.assertEqual(package["registryType"], "oci")
        self.assertEqual(package["runtimeHint"], "docker")
        self.assertEqual(package["transport"], {"type": "stdio"})
        self.assertEqual(
            package["identifier"],
            f"ghcr.io/modelscope/funasr-mcp:{self.metadata['version']}",
        )

    def test_readme_uses_current_published_image(self):
        readme = README.read_text()
        version = self.metadata["version"]

        self.assertIn(f"ghcr.io/modelscope/funasr-mcp:{version}", readme)
        self.assertIn(f"releases/tag/mcp-v{version}", readme)
        self.assertIn("registry.modelcontextprotocol.io", readme)
        if version != "0.1.0":
            self.assertNotIn("ghcr.io/modelscope/funasr-mcp:0.1.0", readme)

    def test_oci_package_mounts_audio_read_only(self):
        package = self.metadata["packages"][0]
        mounts = [
            argument
            for argument in package["runtimeArguments"]
            if argument.get("name") == "--mount"
            and "audio_directory" in argument.get("value", "")
        ]
        self.assertEqual(len(mounts), 1)
        mount = mounts[0]
        self.assertIn("dst=/audio", mount["value"])
        self.assertIn("readonly", mount["value"])
        self.assertEqual(mount["variables"]["audio_directory"]["format"], "filepath")
        self.assertTrue(mount["variables"]["audio_directory"]["isRequired"])

    def test_docker_ownership_label_matches_server_name(self):
        dockerfile = DOCKERFILE.read_text()
        match = re.search(
            r'io\.modelcontextprotocol\.server\.name="([^"]+)"', dockerfile
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), self.metadata["name"])
        self.assertIn("ARG FUNASR_VERSION=1.3.14", dockerfile)
        self.assertIn("funasr==${FUNASR_VERSION}", dockerfile)

    def test_release_workflow_is_versioned_and_oidc_authenticated(self):
        workflow = WORKFLOW.read_text()
        self.assertIn('"mcp-v*"', workflow)
        self.assertIn("MCP_PUBLISHER_VERSION: v1.8.0", workflow)
        self.assertIn(
            "MCP_PUBLISHER_SHA256: 1370446bbe74d562608e8005a6ccce02d146a661fbd78674e11cc70b9618d6cf",
            workflow,
        )
        self.assertIn("id-token: write", workflow)
        self.assertIn("login github-oidc", workflow)
        self.assertIn("python examples/mcp_server/smoke_test.py", workflow)
        self.assertIn("actions/checkout@v7", workflow)
        self.assertIn("actions/setup-python@v6", workflow)
        self.assertIn("docker/build-push-action@v7", workflow)
        self.assertIn("docker/setup-buildx-action@v4", workflow)
        self.assertIn("docker/login-action@v4", workflow)
        self.assertIn("continue-on-error: true", workflow)
        self.assertIn("Verify the versioned image is publicly pullable", workflow)
        self.assertIn("scope=repository:modelscope/funasr-mcp:pull", workflow)
        self.assertIn("https://ghcr.io/v2/modelscope/funasr-mcp/manifests/${IMAGE_VERSION}", workflow)
        self.assertNotIn("actions/checkout@v4", workflow)
        self.assertNotIn("actions/setup-python@v5", workflow)
        self.assertNotIn("docker/build-push-action@v6", workflow)
        self.assertNotIn("docker/setup-buildx-action@v3", workflow)


class MCPContainerSmokeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.smoke_test = load_smoke_test_module()

    def test_validate_responses_accepts_initialize_and_tools_list(self):
        responses = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"serverInfo": {"name": "funasr", "version": "1.3.14"}},
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": [{"name": "transcribe_audio"}]},
            },
        ]

        self.smoke_test.validate_responses(responses)

    def test_validate_responses_rejects_missing_tool(self):
        responses = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"serverInfo": {"name": "funasr", "version": "1.3.14"}},
            },
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}},
        ]

        with self.assertRaisesRegex(ValueError, "transcribe_audio"):
            self.smoke_test.validate_responses(responses)

    def test_validate_responses_rejects_non_object_payload(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            self.smoke_test.validate_responses([{"id": 1}, ["unexpected"]])

    def test_validate_responses_reports_unexpected_ids_without_sorting(self):
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": []}},
            {"jsonrpc": "2.0", "method": "notifications/progress"},
        ]

        with self.assertRaisesRegex(ValueError, "expected response IDs 1 and 2"):
            self.smoke_test.validate_responses(responses)

    def test_parse_responses_rejects_non_json_stdout(self):
        stdout = 'library log on stdout\n{"jsonrpc":"2.0","id":1,"result":{}}\n'

        with self.assertRaisesRegex(ValueError, "non-JSON stdout"):
            self.smoke_test.parse_responses(stdout)


if __name__ == "__main__":
    unittest.main()
