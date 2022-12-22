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
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NOT_QUOTE, DAMO_SIGMA, DAMO_SPACE, GraphFst
from fun_text_processing.text_normalization.es.graph_utils import shift_number_gender
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinals
        e.g. ordinal { integer: "tercer" } } -> "tercero"
                                           -> "tercera"
										   -> "tercer"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="ordinal", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete("integer: \"") + pynini.closure(DAMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        # masculne gender we leave as is
        graph_masc = graph + pynutil.delete(" morphosyntactic_features: \"gender_masc")

        # shift gender
        graph_fem_ending = graph @ pynini.cdrewrite(
            pynini.cross("o", "a"), "", DAMO_SPACE | pynini.accep("[EOS]"), DAMO_SIGMA
        )
        graph_fem = shift_number_gender(graph_fem_ending) + pynutil.delete(" morphosyntactic_features: \"gender_fem")

        # Apocope just changes tercero and primero. May occur if someone wrote 11.er (uncommon)
        graph_apocope = (
            pynini.cross("tercero", "tercer")
            | pynini.cross("primero", "primer")
            | pynini.cross("undécimo", "decimoprimer")
        )  # In case someone wrote 11.er with deterministic
        graph_apocope = (graph @ pynini.cdrewrite(graph_apocope, "", "", DAMO_SIGMA)) + pynutil.delete(
            " morphosyntactic_features: \"apocope"
        )

        graph = graph_apocope | graph_masc | graph_fem

        if not deterministic:
            # Plural graph
            graph_plural = pynini.cdrewrite(
                pynutil.insert("s"), pynini.union("o", "a"), DAMO_SPACE | pynini.accep("[EOS]"), DAMO_SIGMA
            )

            graph |= (graph @ graph_plural) + pynutil.delete("/plural")

        self.graph = graph + pynutil.delete("\"")

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
