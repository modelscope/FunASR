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
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, GraphFst, delete_space, insert_space
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. time { hours: "два часа пятнадцать минут" } -> "два часа пятнадцать минут"

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        minutes = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        self.graph = (
            hour + delete_space + insert_space + minutes + delete_space + pynutil.delete("preserve_order: true")
        )
        self.graph |= hour + delete_space
        self.graph |= minutes + delete_space + insert_space + hour + delete_space

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
