import os
import string
from pathlib import Path
from typing import Dict

import pynini
from pynini import Far
from pynini.export import export
from pynini.lib import byte, pynutil, utf8

FUN_CHAR = utf8.VALID_UTF8_CHAR

FUN_DIGIT = byte.DIGIT
FUN_LOWER = pynini.union(*string.ascii_lowercase).optimize()
FUN_UPPER = pynini.union(*string.ascii_uppercase).optimize()
FUN_ALPHA = pynini.union(FUN_LOWER, FUN_UPPER).optimize()

FUN_SPACE = " "
FUN_WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", "\u00A0").optimize()
FUN_NOT_SPACE = pynini.difference(FUN_CHAR, FUN_WHITE_SPACE).optimize()
FUN_NOT_QUOTE = pynini.difference(FUN_CHAR, r'"').optimize()

FUN_PUNCT = pynini.union(*map(pynini.escape, string.punctuation)).optimize()


FUN_SIGMA = pynini.closure(FUN_CHAR)

delete_space = pynutil.delete(pynini.closure(FUN_WHITE_SPACE))
delete_zero_or_one_space = pynutil.delete(pynini.closure(FUN_WHITE_SPACE, 0, 1))
insert_space = pynutil.insert(" ")
delete_extra_space = pynini.cross(pynini.closure(FUN_WHITE_SPACE, 1), " ")


def generator_main(file_name: str, graphs: Dict[str, "pynini.FstLike"]):
    """
    Exports graph as OpenFst finite state archive (FAR) file with given file name and rule name.

    Args:
        file_name: exported file name
        graphs: Mapping of a rule name and Pynini WFST graph to be exported
    """
    exporter = export.Exporter(file_name)
    for rule, graph in graphs.items():
        exporter[rule] = graph.optimize()
    exporter.close()
    print(f"Created {file_name}")


class GraphFst:
    """
    Base class for all grammar fsts.

    Args:
        name: name of grammar class
        kind: either 'classify' or 'verbalize'
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, name: str, kind: str, deterministic: bool = True):
        self.name = name
        self.kind = kind
        self._fst = None
        self.deterministic = deterministic

        self.far_path = Path(os.path.dirname(__file__) + "/grammars/" + kind + "/" + name + ".far")
        if self.far_exist():
            self._fst = Far(
                self.far_path, mode="r", arc_type="standard", far_type="default"
            ).get_fst()

    def far_exist(self) -> bool:
        """
        Returns true if FAR can be loaded
        """
        return self.far_path.exists()

    @property
    def fst(self) -> "pynini.FstLike":
        return self._fst

    @fst.setter
    def fst(self, fst):
        self._fst = fst

    def add_tokens(self, fst) -> "pynini.FstLike":
        """
        Wraps class name around to given fst

        Args:
            fst: input fst

        Returns:
            Fst: fst
        """
        return pynutil.insert(f"{self.name} {{ ") + fst + pynutil.insert(" }")

    def delete_tokens(self, fst) -> "pynini.FstLike":
        """
        Deletes class name wrap around output of given fst

        Args:
            fst: input fst

        Returns:
            Fst: fst
        """
        res = (
            pynutil.delete(f"{self.name}")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + fst
            + delete_space
            + pynutil.delete("}")
        )
        return res @ pynini.cdrewrite(pynini.cross("\u00A0", " "), "", "", FUN_SIGMA)
