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
from fun_text_processing.text_normalization.zh.graph_utils import (
    FUN_ALPHA,
    FUN_DIGIT,
    FUN_PUNCT,
    FUN_SIGMA,
    FUN_WHITE_SPACE,
    GraphFst,
)
from fun_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil, utf8


class PostProcessor(GraphFst):
    '''
        Postprocessing of TN, now contains:
            1. punctuation removal
            2. letter case conversion
            3. oov tagger
    '''

    def __init__(
        self, remove_puncts: bool = False, to_upper: bool = False, to_lower: bool = False, tag_oov: bool = False,
    ):
        super().__init__(name="PostProcessor", kind="processor")

        graph = pynini.cdrewrite('', '', '', FUN_SIGMA)
        if remove_puncts:
            remove_puncts_graph = pynutil.delete(
                pynini.union(FUN_PUNCT, pynini.string_file(get_abs_path('data/char/punctuations_zh.tsv')))
            )
            graph @= pynini.cdrewrite(remove_puncts_graph, "", "", FUN_SIGMA).optimize()

        if to_upper or to_lower:
            if to_upper:
                conv_cases_graph = pynini.inverse(pynini.string_file(get_abs_path('data/char/upper_to_lower.tsv')))
            else:
                conv_cases_graph = pynini.string_file(get_abs_path('data/char/upper_to_lower.tsv'))

            graph @= pynini.cdrewrite(conv_cases_graph, "", "", FUN_SIGMA).optimize()

        if tag_oov:
            zh_charset_std = pynini.string_file(get_abs_path("data/char/charset_national_standard_2013_8105.tsv"))
            zh_charset_ext = pynini.string_file(get_abs_path("data/char/charset_extension.tsv"))

            zh_charset = (
                zh_charset_std | zh_charset_ext | pynini.string_file(get_abs_path("data/char/punctuations_zh.tsv"))
            )
            en_charset = FUN_DIGIT | FUN_ALPHA | FUN_PUNCT | FUN_WHITE_SPACE
            charset = zh_charset | en_charset

            with open(get_abs_path("data/char/oov_tags.tsv"), "r") as f:
                tags = f.readline().strip().split('\t')
                assert len(tags) == 2
                ltag, rtag = tags

            oov_charset = pynini.difference(utf8.VALID_UTF8_CHAR, charset)
            tag_oov_graph = pynutil.insert(ltag) + oov_charset + pynutil.insert(rtag)
            graph @= pynini.cdrewrite(tag_oov_graph, "", "", FUN_SIGMA).optimize()

        self.fst = graph.optimize()
