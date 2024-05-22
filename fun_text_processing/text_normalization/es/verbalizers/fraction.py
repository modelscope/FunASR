import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_CHAR,
    DAMO_NOT_QUOTE,
    DAMO_NOT_SPACE,
    DAMO_SIGMA,
    DAMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.es.graph_utils import (
    accents,
    shift_cardinal_gender,
    strip_cardinal_apocope,
)
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
            e.g. tokens { fraction { integer: "treinta y tres" numerator: "cuatro" denominator: "quinto" } } ->
        treinta y tres y cuatro quintos


    Args:
            deterministic: if True will provide a single transduction option,
                    for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        # Derivational strings append 'avo' as a suffix. Adding space for processing aid
        fraction_stem = pynutil.insert(" avo")
        plural = pynutil.insert("s")
        conjunction = pynutil.insert(" y ")

        integer = (
            pynutil.delete('integer_part: "')
            + strip_cardinal_apocope(pynini.closure(DAMO_NOT_QUOTE))
            + pynutil.delete('"')
        )

        numerator_one = pynutil.delete('numerator: "') + pynini.accep("un") + pynutil.delete('" ')
        numerator = (
            pynutil.delete('numerator: "')
            + pynini.difference(pynini.closure(DAMO_NOT_QUOTE), "un")
            + pynutil.delete('" ')
        )

        denominator_add_stem = pynutil.delete('denominator: "') + (
            pynini.closure(DAMO_NOT_QUOTE)
            + fraction_stem
            + pynutil.delete('" morphosyntactic_features: "add_root"')
        )
        denominator_ordinal = pynutil.delete('denominator: "') + (
            pynini.closure(DAMO_NOT_QUOTE) + pynutil.delete('" morphosyntactic_features: "ordinal"')
        )
        denominator_cardinal = pynutil.delete('denominator: "') + (
            pynini.closure(DAMO_NOT_QUOTE) + pynutil.delete('"')
        )

        denominator_singular = pynini.union(denominator_add_stem, denominator_ordinal)
        if not deterministic:
            # Occasional exceptions
            denominator_singular |= denominator_add_stem @ pynini.string_map(
                [("once avo", "undécimo"), ("doce avo", "duodécimo")]
            )
        denominator_plural = denominator_singular + plural

        # Merging operations
        merge = pynini.cdrewrite(
            pynini.cross(" y ", "i"), "", "", DAMO_SIGMA
        )  # The denominator must be a single word, with the conjunction "y" replaced by i
        merge @= pynini.cdrewrite(
            delete_space, "", pynini.difference(DAMO_CHAR, "parte"), DAMO_SIGMA
        )

        # The merger can produce duplicate vowels. This is not allowed in orthography
        delete_duplicates = pynini.string_map([("aa", "a"), ("oo", "o")])  # Removes vowels
        delete_duplicates = pynini.cdrewrite(delete_duplicates, "", "", DAMO_SIGMA)

        remove_accents = pynini.cdrewrite(
            accents,
            pynini.union(DAMO_SPACE, pynini.accep("[BOS]")) + pynini.closure(DAMO_NOT_SPACE),
            pynini.closure(DAMO_NOT_SPACE) + pynini.union("avo", "ava", "ésimo", "ésima"),
            DAMO_SIGMA,
        )
        merge_into_single_word = merge @ remove_accents @ delete_duplicates

        fraction_default = (
            numerator + delete_space + insert_space + (denominator_plural @ merge_into_single_word)
        )

        fraction_with_one = (
            numerator_one
            + delete_space
            + insert_space
            + (denominator_singular @ merge_into_single_word)
        )

        fraction_with_cardinal = strip_cardinal_apocope(numerator | numerator_one)
        fraction_with_cardinal += (
            delete_space + pynutil.insert(" sobre ") + strip_cardinal_apocope(denominator_cardinal)
        )

        if not deterministic:
            # There is an alternative rendering where ordinals act as adjectives for 'parte'. This requires use of the feminine
            # Other rules will manage use of "un" at end, so just worry about endings
            exceptions = pynini.string_map([("tercia", "tercera")])
            apply_exceptions = pynini.cdrewrite(exceptions, "", "", DAMO_SIGMA)
            vowel_change = pynini.cdrewrite(
                pynini.cross("o", "a"), "", pynini.accep("[EOS]"), DAMO_SIGMA
            )

            denominator_singular_fem = (
                shift_cardinal_gender(denominator_singular) @ vowel_change @ apply_exceptions
            )
            denominator_plural_fem = denominator_singular_fem + plural

            numerator_one_fem = shift_cardinal_gender(numerator_one)
            numerator_fem = shift_cardinal_gender(numerator)

            fraction_with_cardinal |= (
                (numerator_one_fem | numerator_fem)
                + delete_space
                + pynutil.insert(" sobre ")
                + shift_cardinal_gender(denominator_cardinal)
            )

            # Still need to manage stems
            merge_stem = pynini.cdrewrite(
                delete_space, "", pynini.union("avo", "ava", "avos", "avas"), DAMO_SIGMA
            )  # For managing alternative spacing
            merge_stem @= remove_accents @ delete_duplicates

            fraction_with_one_fem = numerator_one_fem + delete_space + insert_space
            fraction_with_one_fem += pynini.union(
                denominator_singular_fem @ merge_stem,
                denominator_singular_fem @ merge_into_single_word,
            )  # Both forms exists
            fraction_with_one_fem += pynutil.insert(" parte")
            fraction_with_one_fem @= pynini.cdrewrite(
                pynini.cross("una media", "media"), "", "", DAMO_SIGMA
            )  # "media" not "una media"

            fraction_default_fem = numerator_fem + delete_space + insert_space
            fraction_default_fem += pynini.union(
                denominator_plural_fem @ merge_stem, denominator_plural_fem @ merge_into_single_word
            )
            fraction_default_fem += pynutil.insert(" partes")

            fraction_default |= (
                numerator + delete_space + insert_space + denominator_plural @ merge_stem
            )  # Case of no merger
            fraction_default |= fraction_default_fem

            fraction_with_one |= (
                numerator_one + delete_space + insert_space + denominator_singular @ merge_stem
            )
            fraction_with_one |= fraction_with_one_fem

        fraction_with_one @= pynini.cdrewrite(
            pynini.cross("un medio", "medio"), "", "", DAMO_SIGMA
        )  # "medio" not "un medio"

        fraction = fraction_with_one | fraction_default | fraction_with_cardinal
        graph_masc = pynini.closure(integer + delete_space + conjunction, 0, 1) + fraction

        # Manage cases of fem gender (only shows on integer except for "medio")
        integer_fem = shift_cardinal_gender(integer)
        fraction_default |= (
            shift_cardinal_gender(numerator)
            + delete_space
            + insert_space
            + (denominator_plural @ pynini.cross("medios", "medias"))
        )
        fraction_with_one |= (
            pynutil.delete(numerator_one)
            + delete_space
            + (denominator_singular @ pynini.cross("medio", "media"))
        )

        fraction_fem = fraction_with_one | fraction_default | fraction_with_cardinal
        graph_fem = pynini.closure(integer_fem + delete_space + conjunction, 0, 1) + fraction_fem

        self.graph_masc = pynini.optimize(graph_masc)
        self.graph_fem = pynini.optimize(graph_fem)

        self.graph = graph_masc | graph_fem

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
