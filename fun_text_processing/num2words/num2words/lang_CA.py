# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_CA(lang_EU.Num2Word_EU):
    def set_high_numwords(self, high):
        max = 3 + 3 * len(high)
        for word, n in zip(high, range(max, 3, -3)):
            self.cards[10 ** n] = word + "illions"

    def setup(self):
        super(Num2Word_CA, self).setup()

        self.negword = "menys "
        self.pointword = "punt"
        self.exclude_title = ["i", "punt", "menys"]

        self.mid_numwords = [(1000, "mil"), (100, "cent"),
                             (90, "noranta"), (80, "vuitanta"), (70, "setanta"),
                             (60, "seixanta"), (50, "cinquanta"), (40, "quaranta"),
                             (30, "trenta")]
        self.low_numwords = ['vint', 'dinou', 'divuit anys', 'disset', 'setze', 'quinze', 'catorze', 'tretze', 'dotze', 'onze', 'deu', 'nou', 'vuit', 'set', 'sis', 'cinc', 'quatre', 'tres', 'dos', 'un', 'zero']
        self.ords = {'un': 'primer',
                    'dos': 'segon',
                    'tres': 'tercer',
                    'quatre': 'quart',
                    'cinc': 'cinquè',
                    'sis': 'sisè',
                    'set': 'setè',
                    'vuit': 'vuitè',
                    'nou': 'novè',
                    'deu': 'desè'}

    def merge(self, lpair, rpair):
        ltext, lnum = lpair
        rtext, rnum = rpair
        if lnum == 1 and rnum < 100:
            return (rtext, rnum)
        elif 100 > lnum > rnum:
            return ("%s-%s" % (ltext, rtext), lnum + rnum)
        elif lnum >= 100 > rnum:
            return ("%s i %s" % (ltext, rtext), lnum + rnum)
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
            if lastword[-1] == "y":
                lastword = lastword[:-1] + "ie"
            lastword += "th"
        lastwords[-1] = self.title(lastword)
        outwords[-1] = "-".join(lastwords)
        return " ".join(outwords)

    def to_ordinal_num(self, value):
        self.verify_ordinal(value)
        return "%s%s" % (value, self.to_ordinal(value)[-2:])

    def to_year(self, val, suffix=None, longval=True):
        if val < 0:
            val = abs(val)
            suffix = 'BC' if not suffix else suffix
        high, low = (val // 100, val % 100)
        # If year is 00XX, X00X, or beyond 9999, go cardinal.
        if (high == 0
                or (high % 10 == 0 and low < 10)
                or high >= 100):
            valtext = self.to_cardinal(val)
        else:
            hightext = self.to_cardinal(high)
            if low == 0:
                lowtext = "cent"
            elif low < 10:
                lowtext = "oh-%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s %s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s %s" % (valtext, suffix))
