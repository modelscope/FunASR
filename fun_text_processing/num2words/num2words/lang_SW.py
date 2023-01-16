# -*- coding: utf-8 -*-

# Swahili number to words

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_SW(lang_EU.Num2Word_EU):
    # GIGA_SUFFIX = "iljarder"
    # MEGA_SUFFIX = "iljoner"

    # def set_high_numwords(self, high):
    #     cap = 3 + 6 * len(high)

    #     for word, n in zip(high, range(cap, 3, -6)):
    #         if self.GIGA_SUFFIX:
    #             self.cards[10 ** n] = word + self.GIGA_SUFFIX

    #         if self.MEGA_SUFFIX:
    #             self.cards[10 ** (n - 3)] = word + self.MEGA_SUFFIX
    def set_high_numwords(self, high):
        max = 3 * len(high)
        for word, n in zip(high, range(max, 0, -3)):
            # print(word[0],word[1],n)
            self.cards[10 ** n] = word[1]

    def setup(self):
        super(Num2Word_SW, self).setup()

        self.negword = "kuondoa "
        self.pointword = "hatua"
        self.exclude_title = ["na", "hatua", "kuondoa"]

        self.high_numwords = [(10000000000, 'bilioni kumi'),
                              (1000000000, 'bilioni moja'),
                              (100000000, 'milioni mia moja'), 
                              (10000000,'milioni kumi'), 
                              (1000000, 'milioni moja'), 
                              (100000, 'laki moja'), 
                              (1000, 'elfu moja') ]
        self.mid_numwords = [(1000, "elfu moja"), (100, "mia moja"),
                             (90, "tisini"), (80, "themanini"), (70, "sabini"),
                             (60, "sitini"), (50, "hamsini"), (40, "arobaini"),
                             (30, "thelathini")]

        self.low_numwords = ['sufuri', 'moja', 'mbili', 'tatu', 'nne', 'tano', 'sita', 'saba', 'nane', 'tisa', 'kumi', 'kumi na moja', 'kumi na mbili', 'kumi na tatu', 'kumi na nne', 'kumi na tano', 'kumi na sita', 'kumi na saba', 'kumi na nane', 'kumi na tisa', 'ishirini']

        self.ords = {"moja":"kwanza", 
                    "mbili":"pili", 
                    "tatu":"cha tatu", 
                    "nne":"nne", 
                    "tano":"tano", 
                    "sita":"ya sita", 
                    "saba":"ya saba", 
                    "nane":"ya nane", 
                    "tisa":"ya tisa", 
                    "kumi":"ya kumi"}


    def merge(self, lpair, rpair):
        ltext, lnum = lpair
        rtext, rnum = rpair
        if lnum == 1 and rnum < 100:
            return (rtext, rnum)
        elif 100 > lnum > rnum:
            return ("%s %s" % (ltext, rtext), lnum + rnum)
        elif lnum >= 100 > rnum:
            return ("%s %s" % (ltext, rtext), lnum + rnum)
        elif rnum > lnum:
            return ("%s %s" % (ltext, rtext), lnum * rnum)
        return ("%s %s" % (ltext, rtext), lnum + rnum)

    def to_ordinal_num(self, value):
        self.verify_ordinal(value)
        return "%s %s" % (value, self.to_ordinal(value))

    def to_ordinal(self, value):
        self.verify_ordinal(value)
        try:
            ordinal_word = self.ords[value]
        except KeyError:
            ordinal_word = value #TODO: check nepali ordinal word suffix 
        return ordinal_word


    def to_year(self, val, suffix=None, longval=True):
        if val < 0:
            val = abs(val)
            suffix = 'kabla ya karne ' if not suffix else suffix
        high, low = (val // 100, val % 100)
        # If year is 00XX, X00X, or beyond 9999, go cardinal.
        if (high == 0
                or (high % 10 == 0 and low < 10)
                or high >= 100):
            valtext = self.to_cardinal(val)
        else:
            hightext = self.to_cardinal(high)
            if low == 0:
                lowtext = "mia"
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s %s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s %s" % (valtext, suffix))
