import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    MIN_NEG_WEIGHT,
    DAMO_ALPHA,
    DAMO_DIGIT,
    DAMO_NOT_SPACE,
    DAMO_SIGMA,
    GraphFst,
    convert_space,
    get_abs_path,
)
from fun_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from pynini.examples import plurals
from pynini.lib import pynutil


class WordFst(GraphFst):
    """
    Finite state transducer for classifying word. Considers sentence boundary exceptions.
        e.g. sleep -> tokens { name: "sleep" }

    Args:
        punctuation: PunctuationFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, punctuation: GraphFst, deterministic: bool = True):
        super().__init__(name="word", kind="classify", deterministic=deterministic)

        punct = PunctuationFst().graph
        default_graph = pynini.closure(pynini.difference(DAMO_NOT_SPACE, punct.project("input")), 1)
        symbols_to_exclude = (
            pynini.union("$", "€", "₩", "£", "¥", "#", "%") | DAMO_DIGIT
        ).optimize()
        graph = pynini.closure(pynini.difference(DAMO_NOT_SPACE, symbols_to_exclude), 1)
        graph = pynutil.add_weight(graph, MIN_NEG_WEIGHT) | default_graph

        # leave phones of format [HH AH0 L OW1] untouched
        phoneme_unit = pynini.closure(DAMO_ALPHA, 1) + pynini.closure(DAMO_DIGIT)
        phoneme = (
            pynini.accep(pynini.escape("["))
            + pynini.closure(phoneme_unit + pynini.accep(" "))
            + phoneme_unit
            + pynini.accep(pynini.escape("]"))
        )

        # leave IPA phones of format [ˈdoʊv] untouched, single words and sentences with punctuation marks allowed
        punct_marks = pynini.union(*punctuation.punct_marks).optimize()
        stress = pynini.union("ˈ", "'", "ˌ")
        ipa_phoneme_unit = pynini.string_file(get_abs_path("data/whitelist/ipa_symbols.tsv"))
        # word in ipa form
        ipa_phonemes = (
            pynini.closure(stress, 0, 1)
            + pynini.closure(ipa_phoneme_unit, 1)
            + pynini.closure(stress | ipa_phoneme_unit)
        )
        # allow sentences of words in IPA format separated with spaces or punct marks
        delim = (punct_marks | pynini.accep(" ")) ** (1, ...)
        ipa_phonemes = (
            ipa_phonemes + pynini.closure(delim + ipa_phonemes) + pynini.closure(delim, 0, 1)
        )
        ipa_phonemes = (
            pynini.accep(pynini.escape("[")) + ipa_phonemes + pynini.accep(pynini.escape("]"))
        ).optimize()

        if not deterministic:
            phoneme = (
                pynini.accep(pynini.escape("["))
                + pynini.closure(pynini.accep(" "), 0, 1)
                + pynini.closure(phoneme_unit + pynini.accep(" "))
                + phoneme_unit
                + pynini.closure(pynini.accep(" "), 0, 1)
                + pynini.accep(pynini.escape("]"))
            ).optimize()
            ipa_phonemes = (
                pynini.accep(pynini.escape("[")) + ipa_phonemes + pynini.accep(pynini.escape("]"))
            ).optimize()

        phoneme |= ipa_phonemes
        self.graph = plurals._priority_union(convert_space(phoneme.optimize()), graph, DAMO_SIGMA)
        self.fst = (pynutil.insert('name: "') + self.graph + pynutil.insert('"')).optimize()
