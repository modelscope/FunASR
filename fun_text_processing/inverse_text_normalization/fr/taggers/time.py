import pynini
from fun_text_processing.inverse_text_normalization.fr.graph_utils import GraphFst, delete_space
from fun_text_processing.inverse_text_normalization.fr.utils import get_abs_path
from pynini.lib import pynutil


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. huit heures -> time { hours: "8" minutes: "00" }
        e.g. treize heures -> time { hours: "13" minutes: "00" }
        e.g. treize heures dix -> time { hours: "13" minutes: "10" }
        e.g. huit heures du matin -> time { hours: "8" minutes: "00" suffix: "avant mid"}
        e.g. huite heures du après midi -> time { hours: "8" minutes: "00" suffix: "après-midi"}
        e.g. douze heures moins qart -> time { hours: "11" minutes: "45" }
        e.g. douze heures et qart -> time { hours: "12" minutes: "15" }
        e.g. midi et qart -> time { hours: "12" minutes: "15" }
        e.g. minuit et medi -> time { hours: "0" minutes: "30" }
        e.g. douze heures moins medi -> time { hours: "11" minutes: "30" }
        e.g. douze heures moins trois -> time { hours: "11" minutes: "57" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")
        # hours, minutes, seconds, suffix, zone, style, speak_period

        # time_zone = pynini.invert(pynini.string_file(get_abs_path("data/time/time_zone.tsv")))
        graph_hours_to = pynini.string_file(get_abs_path("data/time/hours_to.tsv"))
        graph_minutes_to = pynini.string_file(get_abs_path("data/time/minutes_to.tsv"))
        graph_hours = pynini.string_file(get_abs_path("data/time/hours.tsv"))
        graph_minutes = pynini.string_file(get_abs_path("data/time/minutes.tsv"))
        graph_suffix_am = pynini.string_file(get_abs_path("data/time/time_suffix_am.tsv"))
        graph_suffix_pm = pynini.string_file(get_abs_path("data/time/time_suffix_pm.tsv"))

        graph_suffix = pynini.cross(graph_suffix_am, "am") | pynini.cross(graph_suffix_pm, "pm")

        # Mapping 'heures'
        graph_heures = pynini.accep("heure") + pynini.accep("s").ques
        graph_heures = pynutil.delete(graph_heures)

        graph_hours += delete_space + graph_heures

        # Midi and minuit
        graph_midi = pynini.cross("midi", "12")
        graph_minuit = pynini.cross("minuit", "0")

        # Mapping 'et demi' and 'et qart'
        graph_et = pynutil.delete("et") + delete_space

        graph_demi = pynini.accep("demi")
        graph_demi += pynini.accep("e").ques  # people vary on feminine or masculine form
        graph_demi = pynini.cross(graph_demi, "30")

        graph_quart = pynini.accep("quart")
        graph_quart = pynini.accep("le ").ques + graph_quart  # sometimes used
        graph_quart = pynini.cross(graph_quart, "15")
        graph_trois_quart = pynini.cross("trois quarts", "45")

        graph_fractions = pynini.union(graph_demi, graph_quart, graph_trois_quart)

        graph_et_fractions = graph_et + graph_fractions

        # Hours component is usually just a cardinal + 'heures' (ignored in case of 'midi/minuit').
        graph_hours_component = pynini.union(graph_hours, graph_midi, graph_minuit)
        graph_hours_component = (
            pynutil.insert('hours: "') + graph_hours_component + pynutil.insert('"')
        )
        graph_hours_component += delete_space

        # Minutes component
        graph_minutes_component = (
            pynutil.insert(' minutes: "')
            + pynini.union(graph_minutes, graph_et_fractions)
            + pynutil.insert('"')
        )

        # Hour and minutes together. For 'demi' and 'qart', 'et' is used as a conjunction.
        graph_time_standard = graph_hours_component + graph_minutes_component.ques

        # For time until hour. "quatre heures moins qart" -> 4 h 00 - 0 h 15 = 3 h 45
        graph_moins = pynutil.delete("moins")
        graph_moins += delete_space

        graph_hours_to_component = graph_hours | graph_midi | graph_minuit
        graph_hours_to_component @= graph_hours_to
        graph_hours_to_component = (
            pynutil.insert('hours: "') + graph_hours_to_component + pynutil.insert('"')
        )
        graph_hours_to_component += delete_space

        graph_minutes_to_component = pynini.union(graph_minutes, graph_fractions)
        graph_minutes_to_component @= graph_minutes_to
        graph_minutes_to_component = (
            pynutil.insert(' minutes: "') + graph_minutes_to_component + pynutil.insert('"')
        )

        graph_time_to = graph_hours_to_component + graph_moins + graph_minutes_to_component

        graph_time_no_suffix = graph_time_standard | graph_time_to

        graph_suffix_component = pynutil.insert(' suffix: "') + graph_suffix + pynutil.insert('"')
        graph_suffix_component = delete_space + graph_suffix_component
        graph_suffix_component = graph_suffix_component.ques

        final_graph = graph_time_no_suffix + graph_suffix_component

        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
