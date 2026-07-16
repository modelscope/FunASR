import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "runtime" / "python" / "onnxruntime"
SETUP_PATH = PACKAGE_ROOT / "setup.py"
EXPECTED_VERSION = "0.4.2"


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


class FunASROnnxReleaseContractTest(unittest.TestCase):
    def test_release_version_is_0_4_2(self):
        self.assertEqual(assigned_literal(read_setup_tree(), "VERSION_NUM"), EXPECTED_VERSION)

    def test_runtime_dependencies_keep_onnx_install_torch_free(self):
        requirements = setup_keyword_literal(read_setup_tree(), "install_requires")
        names = {
            requirement.split(";", 1)[0]
            .split("[", 1)[0]
            .split("=", 1)[0]
            .split("<", 1)[0]
            .split(">", 1)[0]
            .strip()
            .lower()
            .replace("_", "-")
            for requirement in requirements
        }
        self.assertIn("jieba", names)
        self.assertNotIn("torch", names)

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


if __name__ == "__main__":
    unittest.main()
