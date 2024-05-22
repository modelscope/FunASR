import pynini
from fun_text_processing.text_normalization.en.graph_utils import DAMO_SIGMA, DAMO_SPACE
from fun_text_processing.text_normalization.es import LOCALIZATION
from fun_text_processing.text_normalization.es.utils import get_abs_path, load_labels
from pynini.lib import pynutil

digits = pynini.project(pynini.string_file(get_abs_path("data/numbers/digit.tsv")), "input")
tens = pynini.project(pynini.string_file(get_abs_path("data/numbers/ties.tsv")), "input")
teens = pynini.project(pynini.string_file(get_abs_path("data/numbers/teen.tsv")), "input")
twenties = pynini.project(pynini.string_file(get_abs_path("data/numbers/twenties.tsv")), "input")
hundreds = pynini.project(pynini.string_file(get_abs_path("data/numbers/hundreds.tsv")), "input")

accents = pynini.string_map([("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")])

if LOCALIZATION == "am":  # Setting localization for central and northern america formatting
    cardinal_separator = pynini.string_map([",", DAMO_SPACE])
    decimal_separator = pynini.accep(".")
else:
    cardinal_separator = pynini.string_map([".", DAMO_SPACE])
    decimal_separator = pynini.accep(",")

ones = pynini.union("un", "ún")
fem_ones = pynini.union(
    pynini.cross("un", "una"), pynini.cross("ún", "una"), pynini.cross("uno", "una")
)
one_to_one_hundred = pynini.union(
    digits, "uno", tens, teens, twenties, tens + pynini.accep(" y ") + digits
)
fem_hundreds = hundreds @ pynini.cdrewrite(pynini.cross("ientos", "ientas"), "", "", DAMO_SIGMA)


def strip_accent(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Converts all accented vowels to non-accented equivalents

    Args:
        fst: Any fst. Composes vowel conversion onto fst's output strings
    """
    return fst @ pynini.cdrewrite(accents, "", "", DAMO_SIGMA)


def shift_cardinal_gender(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Applies gender conversion rules to a cardinal string. These include: rendering all masculine forms of "uno" (including apocopated forms) as "una" and
    Converting all gendered numbers in the hundreds series (200,300,400...) to feminine equivalent (e.g. "doscientos" -> "doscientas"). Conversion only applies
    to value place for <1000 and multiple of 1000. (e.g. "doscientos mil doscientos" -> "doscientas mil doscientas".) For place values greater than the thousands, there
    is no gender shift as the higher powers of ten ("millones", "billones") are masculine nouns and any conversion would be formally
    ungrammatical.
    e.g.
        "doscientos" -> "doscientas"
        "doscientos mil" -> "doscientas mil"
        "doscientos millones" -> "doscientos millones"
        "doscientos mil millones" -> "doscientos mil millones"
        "doscientos millones doscientos mil doscientos" -> "doscientos millones doscientas mil doscientas"

    Args:
        fst: Any fst. Composes conversion onto fst's output strings
    """
    before_mil = (
        DAMO_SPACE
        + (pynini.accep("mil") | pynini.accep("milésimo"))
        + pynini.closure(DAMO_SPACE + hundreds, 0, 1)
        + pynini.closure(DAMO_SPACE + one_to_one_hundred, 0, 1)
        + pynini.union(pynini.accep("[EOS]"), pynini.accep('"'), decimal_separator)
    )
    before_double_digits = pynini.closure(DAMO_SPACE + one_to_one_hundred, 0, 1) + pynini.union(
        pynini.accep("[EOS]"), pynini.accep('"')
    )

    fem_allign = pynini.cdrewrite(
        fem_hundreds, "", before_mil, DAMO_SIGMA
    )  # doscientas mil dosciento
    fem_allign @= pynini.cdrewrite(
        fem_hundreds, "", before_double_digits, DAMO_SIGMA
    )  # doscientas mil doscienta

    fem_allign @= pynini.cdrewrite(
        fem_ones, "", pynini.union("[EOS]", '"', decimal_separator), DAMO_SIGMA
    )  # If before a quote or EOS, we know it's the end of a string

    return fst @ fem_allign


def shift_number_gender(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Performs gender conversion on all verbalized numbers in output. All values in the hundreds series (200,300,400) are changed to
    feminine gender (e.g. "doscientos" -> "doscientas") and all forms of "uno" (including apocopated forms) are converted to "una".
    This has no boundary restriction and will perform shift across all values in output string.
    e.g.
        "doscientos" -> "doscientas"
        "doscientos millones" -> "doscientas millones"
        "doscientos millones doscientos" -> "doscientas millones doscientas"

    Args:
        fst: Any fst. Composes conversion onto fst's output strings
    """
    fem_allign = pynini.cdrewrite(fem_hundreds, "", "", DAMO_SIGMA)
    fem_allign @= pynini.cdrewrite(
        fem_ones, "", pynini.union(DAMO_SPACE, pynini.accep("[EOS]"), pynini.accep('"')), DAMO_SIGMA
    )  # If before a quote or EOS, we know it's the end of a string

    return fst @ fem_allign


def strip_cardinal_apocope(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Reverts apocope on cardinal strings in line with formation rules. e.g. "un" -> "uno". Due to cardinal formation rules, this in effect only
    affects strings where the final value is a variation of "un".
    e.g.
        "un" -> "uno"
        "veintiún" -> "veintiuno"

    Args:
        fst: Any fst. Composes conversion onto fst's output strings
    """
    # Since cardinals use apocope by default for large values (e.g. "millón"), this only needs to act on the last instance of one
    strip = pynini.cross("un", "uno") | pynini.cross("ún", "uno")
    strip = pynini.cdrewrite(strip, "", pynini.union("[EOS]", '"'), DAMO_SIGMA)
    return fst @ strip


def add_cardinal_apocope_fem(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Adds apocope on cardinal strings in line with stressing rules. e.g. "una" -> "un". This only occurs when "una" precedes a stressed "a" sound in formal speech. This is not predictable
    with text string, so is included for non-deterministic cases.
    e.g.
        "una" -> "un"
        "veintiuna" -> "veintiun"

    Args:
        fst: Any fst. Composes conversion onto fst's output strings
    """
    # Since the stress trigger follows the cardinal string and only affects the preceding sound, this only needs to act on the last instance of one
    strip = pynini.cross("una", "un") | pynini.cross("veintiuna", "veintiún")
    strip = pynini.cdrewrite(strip, "", pynini.union("[EOS]", '"'), DAMO_SIGMA)
    return fst @ strip


def roman_to_int(fst: "pynini.FstLike") -> "pynini.FstLike":
    """
    Alters given fst to convert Roman integers (lower and upper cased) into Arabic numerals. Valid for values up to 1000.
    e.g.
        "V" -> "5"
        "i" -> "1"

    Args:
        fst: Any fst. Composes fst onto Roman conversion outputs.
    """

    def _load_roman(file: str):
        roman = load_labels(get_abs_path(file))
        roman_numerals = [(x, y) for x, y in roman] + [(x.upper(), y) for x, y in roman]
        return pynini.string_map(roman_numerals)

    digit = _load_roman("data/roman/digit.tsv")
    ties = _load_roman("data/roman/ties.tsv")
    hundreds = _load_roman("data/roman/hundreds.tsv")

    graph = (
        digit
        | ties + (digit | pynutil.add_weight(pynutil.insert("0"), 0.01))
        | (
            hundreds
            + (ties | pynutil.add_weight(pynutil.insert("0"), 0.01))
            + (digit | pynutil.add_weight(pynutil.insert("0"), 0.01))
        )
    ).optimize()

    return graph @ fst
