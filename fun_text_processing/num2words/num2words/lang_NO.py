# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_NO(lang_EU.Num2Word_EU):
    GIGA_SUFFIX = "illard"
    MEGA_SUFFIX = "illion"

    def set_high_numwords(self, high):
        cap = 3 + 6 * len(high)

        for word, n in zip(high, range(cap, 3, -6)):
            if self.GIGA_SUFFIX:
                self.cards[10 ** n] = word + self.GIGA_SUFFIX

            if self.MEGA_SUFFIX:
                self.cards[10 ** (n - 3)] = word + self.MEGA_SUFFIX

    def setup(self):
        super(Num2Word_NO, self).setup()

        self.negword = "minus "
        self.pointword = "komma"
        self.exclude_title = ["og", "komma", "minus"]

        self.mid_numwords = [(1000, "tusen"), (100, "hundre"),
                             (90, "nitti"), (80, "åtti"), (70, "sytti"),
                             (60, "seksti"), (50, "femti"), (40, "førti"),
                             (30, "tretti")]
        self.low_numwords = ["tjue", "nitten", "atten", "sytten", "seksten", "femten", "fjorten", "tretten", "tolv", "elleve", "ti", "ni", "åtte", "syv", "seks", "fem", "fire", "tre", "to", "en", "null"]
        self.ords = {"en": "først",
                     "to": "andre",
                     "tre": "tredje",
                     "fire": "fjerde",
                     "fem": "femte",
                     "seks": "sjette",
                     "syv": "syvende",
                     "åtte": "åttende",
                     "ni": "niende",
                     "ti": "tiende",
                     "elleve": "ellevte",
                     "tolv": "tolvte",
                     "tjue": "tjuende"}

    def merge(self, lpair, rpair):
        ltext, lnum = lpair
        rtext, rnum = rpair
        if lnum == 1 and rnum < 100:
            return (rtext, rnum)
        elif 100 > lnum > rnum:
            return ("%s-%s" % (ltext, rtext), lnum + rnum)
        elif lnum >= 100 > rnum:
            return ("%s og %s" % (ltext, rtext), lnum + rnum)
        elif rnum > lnum:
            return ("%s %s" % (ltext, rtext), lnum * rnum)
        return ("%s, %s" % (ltext, rtext), lnum + rnum)

    def to_ordinal(self, value):
        self.verify_ordinal(value)
        outwords = self.to_cardinal(value).split(" ")
        lastwords = outwords[-1].split("-")
        lastword = lastwords[-1].lower()
        try:
            lastword = self.ords[lastword]
        except KeyError:
            if lastword[-2:] == "ti":
                lastword = lastword + "ende"
            else:
                lastword += "de"
        lastwords[-1] = self.title(lastword)
        outwords[-1] = "".join(lastwords)
        return " ".join(outwords)

    def to_ordinal_num(self, value):
        self.verify_ordinal(value)
        return "%s%s" % (value, self.to_ordinal(value)[-2:])

    def to_year(self, val, longval=True):
        if not (val // 100) % 10:
            return self.to_cardinal(val)
        return self.to_splitnum(val, hightxt="hundre", jointxt="og",
                                longval=longval)

    def to_currency(self, val, longval=True):
        return self.to_splitnum(val, hightxt="krone/r", lowtxt="\xf8re/r",
                                jointxt="og", longval=longval, cents=True)
