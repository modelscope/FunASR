import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "industrial_data_pretraining" / "qwen3_asr"
SERVER_PATH = EXAMPLE_DIR / "serve_qwen3_asr_ws.py"
NOTE_PATHS = [
    EXAMPLE_DIR / "serve_qwen3_asr_ws_notes.md",
    EXAMPLE_DIR / "serve_qwen3_asr_ws_notes_en.md",
]


class Qwen3AsrWebsocketExampleTest(unittest.TestCase):
    def test_notes_do_not_contain_review_placeholders(self):
        forbidden_phrases = ["以下是我的个人理解", "The following is my personal understanding"]
        for note_path in NOTE_PATHS:
            text = note_path.read_text(encoding="utf-8")
            for phrase in forbidden_phrases:
                with self.subTest(path=note_path.name, phrase=phrase):
                    self.assertNotIn(phrase, text)

    def test_handler_logs_unexpected_exceptions(self):
        tree = ast.parse(SERVER_PATH.read_text(encoding="utf-8"), filename=str(SERVER_PATH))
        handle_client = next(
            node
            for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_client"
        )

        catches_generic_exception = False
        logs_exception = False
        for node in ast.walk(handle_client):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if isinstance(node.type, ast.Name) and node.type.id == "Exception":
                catches_generic_exception = True
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "logging"
                    and func.attr == "exception"
                ):
                    logs_exception = True

        self.assertTrue(catches_generic_exception)
        self.assertTrue(logs_exception)


if __name__ == "__main__":
    unittest.main()
