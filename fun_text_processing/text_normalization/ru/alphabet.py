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

# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.

import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_NON_BREAKING_SPACE, DAMO_SPACE
from fun_text_processing.text_normalization.ru.utils import get_abs_path

RU_LOWER_ALPHA = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
RU_UPPER_ALPHA = RU_LOWER_ALPHA.upper()
RU_LOWER_ALPHA = pynini.union(*RU_LOWER_ALPHA).optimize()
RU_UPPER_ALPHA = pynini.union(*RU_UPPER_ALPHA).optimize()
RU_ALPHA = (RU_LOWER_ALPHA | RU_UPPER_ALPHA).optimize()

RU_STRESSED_MAP = [
    ("А́", "А'"),
    ("Е́", "Е'"),
    ("Ё́", "Е'"),
    ("И́", "И'"),
    ("О́", "О'"),
    ("У́", "У'"),
    ("Ы́", "Ы'"),
    ("Э́", "Э'"),
    ("Ю́", "Ю'"),
    ("Я́", "Я'"),
    ("а́", "а'"),
    ("е́", "е'"),
    ("ё́", "е'"),
    ("и́", "и'"),
    ("о́", "о'"),
    ("у́", "у'"),
    ("ы́", "ы'"),
    ("э́", "э'"),
    ("ю́", "ю'"),
    ("я́", "я'"),
    ("ё", "е"),
    ("Ё", "Е"),
]

REWRITE_STRESSED = pynini.closure(pynini.string_map(RU_STRESSED_MAP).optimize() | RU_ALPHA).optimize()
TO_CYRILLIC = pynini.string_file(get_abs_path("data/latin_to_cyrillic.tsv")).optimize()
TO_LATIN = pynini.invert(TO_CYRILLIC).optimize()
RU_ALPHA_OR_SPACE = pynini.union(RU_ALPHA, DAMO_SPACE, DAMO_NON_BREAKING_SPACE).optimize()
