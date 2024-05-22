import pynini
from fun_text_processing.text_normalization.en.graph_utils import (
    DAMO_NOT_QUOTE,
    DAMO_SIGMA,
    GraphFst,
    delete_preserve_order,
    delete_space,
    insert_space,
)
from fun_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

alt_minutes = pynini.string_file(get_abs_path("data/time/alt_minutes.tsv"))

morning_times = pynini.string_file(get_abs_path("data/time/morning_times.tsv"))
afternoon_times = pynini.string_file(get_abs_path("data/time/afternoon_times.tsv"))
evening_times = pynini.string_file(get_abs_path("data/time/evening_times.tsv"))


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "doce" minutes: "media" suffix: "a m" } -> doce y media de la noche
        time { hours: "doce" } -> twelve o'clock

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        change_minutes = pynini.cdrewrite(
            alt_minutes, pynini.accep("[BOS]"), pynini.accep("[EOS]"), DAMO_SIGMA
        )

        morning_phrases = pynini.cross("am", "de la mañana")
        afternoon_phrases = pynini.cross("pm", "de la tarde")
        evening_phrases = pynini.cross("pm", "de la noche")

        # For the 12's
        mid_times = pynini.accep("doce")
        mid_phrases = (
            pynini.string_map([("pm", "del mediodía"), ("am", "de la noche")])
            if deterministic
            else pynini.string_map(
                [
                    ("pm", "de la mañana"),
                    ("pm", "del día"),
                    ("pm", "del mediodía"),
                    ("am", "de la noche"),
                    ("am", "de la medianoche"),
                ]
            )
        )

        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        minute = (
            (minute @ change_minutes)
            if deterministic
            else pynini.union(minute, minute @ change_minutes)
        )

        suffix = (
            pynutil.delete("suffix:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        zone = (
            pynutil.delete("zone:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_zone = pynini.closure(delete_space + insert_space + zone, 0, 1)
        second = (
            pynutil.delete("seconds:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(DAMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        graph_hms = (
            hour
            + pynutil.insert(" horas ")
            + delete_space
            + minute
            + pynutil.insert(" minutos y ")
            + delete_space
            + second
            + pynutil.insert(" segundos")
        )

        graph_hm = hour + delete_space + pynutil.insert(" y ") + minute
        graph_hm |= pynini.union(
            (hour @ morning_times)
            + delete_space
            + pynutil.insert(" y ")
            + minute
            + delete_space
            + insert_space
            + (suffix @ morning_phrases),
            (hour @ afternoon_times)
            + delete_space
            + pynutil.insert(" y ")
            + minute
            + delete_space
            + insert_space
            + (suffix @ afternoon_phrases),
            (hour @ evening_times)
            + delete_space
            + pynutil.insert(" y ")
            + minute
            + delete_space
            + insert_space
            + (suffix @ evening_phrases),
            (hour @ mid_times)
            + delete_space
            + pynutil.insert(" y ")
            + minute
            + delete_space
            + insert_space
            + (suffix @ mid_phrases),
        )

        graph_h = pynini.union(
            hour,
            (hour @ morning_times) + delete_space + insert_space + (suffix @ morning_phrases),
            (hour @ afternoon_times) + delete_space + insert_space + (suffix @ afternoon_phrases),
            (hour @ evening_times) + delete_space + insert_space + (suffix @ evening_phrases),
            (hour @ mid_times) + delete_space + insert_space + (suffix @ mid_phrases),
        )

        graph = (graph_hms | graph_hm | graph_h) + optional_zone

        if not deterministic:
            graph_style_1 = pynutil.delete(' style: "1"')
            graph_style_2 = pynutil.delete(' style: "2"')

            graph_menos = hour + delete_space + pynutil.insert(" menos ") + minute + graph_style_1
            graph_menos |= (
                (hour @ morning_times)
                + delete_space
                + pynutil.insert(" menos ")
                + minute
                + delete_space
                + insert_space
                + (suffix @ morning_phrases)
                + graph_style_1
            )
            graph_menos |= (
                (hour @ afternoon_times)
                + delete_space
                + pynutil.insert(" menos ")
                + minute
                + delete_space
                + insert_space
                + (suffix @ afternoon_phrases)
                + graph_style_1
            )
            graph_menos |= (
                (hour @ evening_times)
                + delete_space
                + pynutil.insert(" menos ")
                + minute
                + delete_space
                + insert_space
                + (suffix @ evening_phrases)
                + graph_style_1
            )
            graph_menos |= (
                (hour @ mid_times)
                + delete_space
                + pynutil.insert(" menos ")
                + minute
                + delete_space
                + insert_space
                + (suffix @ mid_phrases)
                + graph_style_1
            )
            graph_menos += optional_zone

            graph_para = minute + pynutil.insert(" para las ") + delete_space + hour + graph_style_2
            graph_para |= (
                minute
                + pynutil.insert(" para las ")
                + delete_space
                + (hour @ morning_times)
                + delete_space
                + insert_space
                + (suffix @ morning_phrases)
                + graph_style_2
            )
            graph_para |= (
                minute
                + pynutil.insert(" para las ")
                + delete_space
                + (hour @ afternoon_times)
                + delete_space
                + insert_space
                + (suffix @ afternoon_phrases)
                + graph_style_2
            )
            graph_para |= (
                minute
                + pynutil.insert(" para las ")
                + delete_space
                + (hour @ evening_times)
                + delete_space
                + insert_space
                + (suffix @ evening_phrases)
                + graph_style_2
            )
            graph_para |= (
                minute
                + pynutil.insert(" para las ")
                + delete_space
                + (hour @ mid_times)
                + delete_space
                + insert_space
                + (suffix @ mid_phrases)
                + graph_style_2
            )
            graph_para += optional_zone
            graph_para @= pynini.cdrewrite(
                pynini.cross(" las ", " la "), "para", "una", DAMO_SIGMA
            )  # Need agreement with one

            graph |= graph_menos | graph_para
        delete_tokens = self.delete_tokens(graph + delete_preserve_order)
        self.fst = delete_tokens.optimize()
