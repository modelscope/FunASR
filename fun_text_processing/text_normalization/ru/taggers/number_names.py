# Copyright 2017 Google Inc.


# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.
#
# Supports cardinals and ordinals in all inflected forms.
#
# The language-specific acceptor G was compiled with digit, teen, decade,
# century, and big power-of-ten preterminals. The lexicon transducer is
# highly ambiguous, but no LM is used.

# Intersects the universal factorization transducer (F) with language-specific
# acceptor (G).

import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_SIGMA
from fun_text_processing.text_normalization.ru.utils import get_abs_path, load_labels
from pynini.lib import pynutil, rewrite


def get_number_names():
    """
    Creates numbers names.

    Based on: 1) Gorman, K., and Sproat, R. 2016. Minimally supervised number normalization.
    Transactions of the Association for Computational Linguistics 4: 507-519.
    and 2) Ng, A. H., Gorman, K., and Sproat, R. 2017.
    Minimally supervised written-to-spoken text normalization. In ASRU, pages 665-670.
    """
    a = pynini.Far(get_abs_path("data/utils/util_arithmetic.far"), mode="r")
    d = a["DELTA_STAR"]
    f = a["IARITHMETIC_RESTRICTED"]
    g = pynini.Fst.read(get_abs_path("data/utils/g.fst"))
    fg = (d @ (f @ (f @ (f @ g).optimize()).optimize()).optimize()).optimize()
    assert rewrite.top_rewrite("230", fg) == "(+ 200 30 +)"

    # Compiles lexicon transducers (L).
    cardinal_name_nominative = pynini.string_file(
        get_abs_path("data/numbers/1_cardinals_nominative.tsv")
    ).optimize()
    cardinal_name_genitive = pynini.string_file(
        get_abs_path("data/numbers/2_cardinals_genitive.tsv")
    ).optimize()
    cardinal_name_dative = pynini.string_file(
        get_abs_path("data/numbers/3_cardinals_dative.tsv")
    ).optimize()
    cardinal_name_accusative = pynini.string_file(
        get_abs_path("data/numbers/4_cardinals_accusative.tsv")
    ).optimize()
    cardinal_name_instrumental = pynini.string_file(
        get_abs_path("data/numbers/5_cardinals_instrumental.tsv")
    ).optimize()
    cardinal_name_prepositional = pynini.string_file(
        get_abs_path("data/numbers/6_cardinals_prepositional.tsv")
    ).optimize()

    cardinal_name_nominative = (
        pynini.closure(cardinal_name_nominative + pynini.accep(" ")) + cardinal_name_nominative
    ).optimize()
    cardinal_l = pynutil.add_weight(cardinal_name_nominative, -0.1)
    for case in [
        pynutil.add_weight(cardinal_name_genitive, 0.1).optimize(),
        pynutil.add_weight(cardinal_name_dative, 0.1).optimize(),
        pynutil.add_weight(cardinal_name_accusative, 0.1).optimize(),
        pynutil.add_weight(cardinal_name_instrumental, 0.1).optimize(),
        pynutil.add_weight(cardinal_name_prepositional, 0.1).optimize(),
    ]:
        cardinal_l |= (pynini.closure(case + pynini.accep(" ")) + case).optimize()

    # Numbers in nominative case (to use, for example, with telephone or serial_graph (in cardinals))
    cardinal_names_nominative_l = (
        pynini.closure(cardinal_name_nominative + pynini.accep(" ")) + cardinal_name_nominative
    ).optimize()

    # Convert e.g. "(* 5 1000 *)" back to  "5000" so complex ordinals will be formed correctly,
    #  e.g. "пятитысячный" will eventually be formed. (If we didn't do this, the incorrect phrase
    # "пять тысячный" would be formed).
    # We do this for all thousands from "(*2 1000 *)" —> "2000" to "(*20 1000 *)" —> "20000".
    # We do not go higher, in order to prevent the WFST graph becoming even larger.
    complex_numbers = pynini.cross("(* 2 1000 *)", "2000")
    for number in range(3, 21):
        complex_numbers |= pynini.cross(f"(* {number} 1000 *)", f"{number}000")

    complex_numbers = (
        DAMO_SIGMA
        + pynutil.add_weight(complex_numbers, -1)
        + pynini.closure(pynini.union(" ", ")", "(", "+", "*"))
    ).optimize()
    fg_ordinal = pynutil.add_weight(pynini.compose(fg, complex_numbers), -1) | fg
    ordinal_name = pynini.string_file(get_abs_path("data/numbers/ordinals.tsv"))
    ordinal_l = (
        pynini.closure(cardinal_name_nominative + pynini.accep(" ")) + ordinal_name
    ).optimize()

    # Composes L with the leaf transducer (P), then composes that with FG.
    p = a["LEAVES"]
    number_names = {}
    number_names["ordinal_number_names"] = (fg_ordinal @ (p @ ordinal_l)).optimize()
    number_names["cardinal_number_names"] = (fg @ (p @ cardinal_l)).optimize()
    number_names["cardinal_names_nominative"] = (fg @ (p @ cardinal_names_nominative_l)).optimize()
    return number_names


def get_alternative_formats():
    """
    Utils to get alternative formats for numbers.
    """
    one_alternatives = load_labels(get_abs_path("data/numbers/cardinals_alternatives.tsv"))
    one_thousand_map = []
    for k in one_alternatives:
        default, alternative = k
        one_thousand_map.append((alternative.split()[1], alternative))
    one_thousand_map = pynini.string_map(one_thousand_map)

    one_thousand_alternative = pynini.cdrewrite(one_thousand_map, "[BOS]", "", DAMO_SIGMA)

    # Adapted from
    # https://github.com/google/TextNormalizationCoveringGrammars/blob/master/src/universal/thousands_punct.grm
    # Specifies common ways of delimiting thousands in digit strings.
    t = pynini.Far(get_abs_path("data/utils/universal_thousands_punct.far"))
    separators = (
        pynutil.add_weight(t["dot_thousands"], 0.1)
        | pynutil.add_weight(t["no_delimiter"], -0.1)
        | pynutil.add_weight(t["space_thousands"], 0.1)
    )
    alternative_formats = {}
    alternative_formats["one_thousand_alternative"] = one_thousand_alternative.optimize()
    alternative_formats["separators"] = separators.optimize()
    return alternative_formats


if __name__ == "__main__":
    from fun_text_processing.text_normalization.en.graph_utils import generator_main

    numbers = get_number_names()
    for k, v in numbers.items():
        generator_main(f"{k}.far", {k: v})
