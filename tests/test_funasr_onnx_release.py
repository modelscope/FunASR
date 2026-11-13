import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "runtime" / "python" / "onnxruntime"
SETUP_PATH = PACKAGE_ROOT / "setup.py"
EXPECTED_VERSION = "0.4.11"
REQUIREMENT_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")


def read_setup_tree():
    return ast.parse(SETUP_PATH.read_text(encoding="utf-8"), filename=str(SETUP_PATH))


def assigned_literal(tree, name):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in {SETUP_PATH}")


def setup_keyword_literal(tree, name):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "setup":
            continue
        for keyword in node.keywords:
            if keyword.arg == name:
                return ast.literal_eval(keyword.value)
    raise AssertionError(f"setuptools.setup() has no {name!r} keyword")


def normalized_requirement_name(requirement):
    match = REQUIREMENT_NAME_PATTERN.match(requirement)
    if match is None:
        raise AssertionError(f"Unable to parse requirement name from {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


class FunASROnnxReleaseContractTest(unittest.TestCase):
    def test_release_version_is_0_4_11(self):
        self.assertEqual(assigned_literal(read_setup_tree(), "VERSION_NUM"), EXPECTED_VERSION)

    def test_standalone_license_preserves_repository_notice(self):
        license_path = PACKAGE_ROOT / "LICENSE"
        self.assertTrue(license_path.is_file())
        self.assertEqual(license_path.read_bytes(), (ROOT / "LICENSE").read_bytes())
        self.assertEqual(setup_keyword_literal(read_setup_tree(), "license_files"), ["LICENSE"])

    def test_workflow_checks_and_artifact_use_release_version(self):
        workflow = (ROOT / ".github/workflows/test-funasr-onnx-package.yml").read_text()
        self.assertIn(f'assert version("funasr-onnx") == "{EXPECTED_VERSION}"', workflow)
        self.assertIn(f"name: funasr-onnx-{EXPECTED_VERSION}", workflow)

    def test_workflow_collects_empty_punctuation_regression(self):
        workflow = (ROOT / ".github/workflows/test-funasr-onnx-package.yml").read_text()
        test_path = "tests/test_funasr_onnx_punctuation_empty_input.py"
        self.assertEqual(workflow.count(test_path), 3)
        self.assertIn("python " + test_path, workflow)

    def test_readme_names_the_standalone_fix_release(self):
        readme = (PACKAGE_ROOT / "README.md").read_text()
        self.assertIn(f'pip install -U "funasr-onnx>={EXPECTED_VERSION}"', readme)
        self.assertIn("does not update the installed ONNX wrappers", readme)
        self.assertIn("#3865", readme)
        self.assertIn("without creating a WAV sidecar", readme)
        self.assertIn("short/final VAD", readme)
        self.assertIn("zero-overlap", readme)
        self.assertIn("one independent result per input", readme)
        self.assertIn("effective model batch remains 1", readme)
        self.assertIn("#3831", readme)
        self.assertIn("#3838", readme)
        self.assertIn("hotword", readme)
        self.assertIn("#3842", readme)
        self.assertIn("CMVN", readme)
        self.assertIn("#3846", readme)
        self.assertIn("#3847", readme)
        self.assertIn("empty or sub-frame", readme)
        self.assertIn("preserves the streaming cache", readme)
        self.assertIn("#3869", readme)
        self.assertIn("#3871", readme)
        self.assertIn("#3882", readme)

    def test_runtime_dependencies_keep_onnx_install_torch_free(self):
        requirements = setup_keyword_literal(read_setup_tree(), "install_requires")
        names = {normalized_requirement_name(requirement) for requirement in requirements}
        self.assertIn("jieba", names)
        self.assertNotIn("torch", names)

    def test_requirement_name_parser_handles_pep508_forms(self):
        cases = {
            "torch!=2.0": "torch",
            "Torch_CUDA~=2.0": "torch-cuda",
            'torch[distributed]>=2.0; python_version >= "3.11"': "torch",
            "torch @ https://example.invalid/torch.whl": "torch",
        }
        for requirement, expected in cases.items():
            with self.subTest(requirement=requirement):
                self.assertEqual(normalized_requirement_name(requirement), expected)

    def test_package_source_has_no_torch_imports(self):
        offenders = []
        package_dir = PACKAGE_ROOT / "funasr_onnx"
        for source_path in sorted(package_dir.rglob("*.py")):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    continue
                if any(module == "torch" or module.startswith("torch.") for module in modules):
                    offenders.append(f"{source_path.relative_to(ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_onnxruntime_import_failure_preserves_the_original_cause(self):
        source_path = PACKAGE_ROOT / "funasr_onnx" / "utils" / "utils.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

        onnx_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.Try)
            and any(
                isinstance(statement, ast.ImportFrom) and statement.module == "onnxruntime"
                for statement in node.body
            )
        ]
        self.assertEqual(len(onnx_imports), 1)

        handlers = onnx_imports[0].handlers
        self.assertEqual(len(handlers), 1)
        handler = handlers[0]
        self.assertIsInstance(handler.type, ast.Name)
        self.assertEqual(handler.type.id, "ImportError")
        self.assertIsNotNone(handler.name)
        self.assertEqual(len(handler.body), 1)

        raised = handler.body[0]
        self.assertIsInstance(raised, ast.Raise)
        self.assertIsInstance(raised.exc, ast.Call)
        self.assertIsInstance(raised.exc.func, ast.Name)
        self.assertEqual(raised.exc.func.id, "ImportError")
        self.assertIsInstance(raised.cause, ast.Name)
        self.assertEqual(raised.cause.id, handler.name)


if __name__ == "__main__":
    unittest.main()

