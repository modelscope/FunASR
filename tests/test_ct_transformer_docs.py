"""Check punctuation API guidance without importing the model or its dependencies."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CTTransformerDocsContract(unittest.TestCase):
    def test_unpunctuated_sensevoice_output_can_use_punctuation_model(self):
        path = ROOT / "funasr/models/ct_transformer/model.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        model = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "CTTransformer")
        doc = " ".join(ast.get_docstring(model).split())
        self.assertIn("unpunctuated SenseVoiceSmall", doc)
        self.assertIn("configure punc_model", doc)
        self.assertIn("when punctuation is needed", doc)
        self.assertNotIn("Only required for Paraformer", doc)


if __name__ == "__main__":
    unittest.main()
