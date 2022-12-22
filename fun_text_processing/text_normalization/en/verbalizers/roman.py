# Copyright NeMo (https://github.com/NVIDIA/NeMo). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst
from fun_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst
from pynini.lib import pynutil


class RomanFst(GraphFst):
    """
    Finite state transducer for verbalizing roman numerals
        e.g. tokens { roman { integer: "one" } } -> one

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="roman", kind="verbalize", deterministic=deterministic)
        suffix = OrdinalFst().suffix

        cardinal = pynini.closure(DAMO_NOT_QUOTE)
        ordinal = pynini.compose(cardinal, suffix)

        graph = (
            pynutil.delete("key_cardinal: \"")
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + pynini.accep(" ")
            + pynutil.delete("integer: \"")
            + cardinal
            + pynutil.delete("\"")
        ).optimize()

        graph |= (
            pynutil.delete("default_cardinal: \"default\" integer: \"") + cardinal + pynutil.delete("\"")
        ).optimize()

        graph |= (
            pynutil.delete("default_ordinal: \"default\" integer: \"") + ordinal + pynutil.delete("\"")
        ).optimize()

        graph |= (
            pynutil.delete("key_the_ordinal: \"")
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + pynini.accep(" ")
            + pynutil.delete("integer: \"")
            + pynini.closure(pynutil.insert("the "), 0, 1)
            + ordinal
            + pynutil.delete("\"")
        ).optimize()

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
