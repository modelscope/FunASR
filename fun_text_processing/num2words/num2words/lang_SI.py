# -*- coding: utf-8 -*-

# Sinhala number to words

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_SI(lang_EU.Num2Word_EU):
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
        super(Num2Word_SI, self).setup()

        self.negword = "අඩු "
        self.pointword = "ලක්ෂ්යය"
        self.exclude_title = ["හා", "ලක්ෂ්යය", "අඩු"]

        self.high_numwords = [(10000000000, 'බිලියන දහයකි'),
                              (1000000000, 'බිලියනයක්'),
                              (100000000, 'මිලියන සියයක්'), 
                              (10000000,'දස මිලියන'), 
                              (1000000, 'මිලියනයක්'), 
                              (100000, 'එක් ලක්ෂයක්'), 
                              (1000, 'දහසක්') ]
        self.mid_numwords = [(1000, "දහසක්"), (100, "සියය"),
                             (90, "අනූවක්"), (80, "අසූව"), (70, "හැත්තෑ"),
                             (60, "හැට"), (50, "පනස්"), (40, "හතළිහක්"),
                             (30, "තිස්")]

        self.low_numwords = ['විස්සක්','දහනවය','දහඅට','දාහත','දහසය','පහළොව','දහහතර','දහතුන','දොළොස්','එකොළොස්','දස','නවය','අට','හත','හය','පහ','හතර','තුන්','දෙක','එක','ශුන්ය']

        self.ords = {'එක': 'පළමුවන',
                    'දෙක': 'දෙවැනි',
                    'තුන්': 'තෙවන',
                    'හතර': 'හතරවන',
                    'පහ': 'පස්වන',
                    'හය': 'හය වන',
                    'හත': 'හත්වන',
                    'අට': 'අටවැනි',
                    'නවය': 'නවවැනි',
                    'දස': 'දහවන'}


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
            suffix = 'සියවසට පෙර ' if not suffix else suffix
        high, low = (val // 100, val % 100)
        # If year is 00XX, X00X, or beyond 9999, go cardinal.
        if (high == 0
                or (high % 10 == 0 and low < 10)
                or high >= 100):
            valtext = self.to_cardinal(val)
        else:
            hightext = self.to_cardinal(high)
            if low == 0:
                lowtext = "සියයක්" #hundred
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s %s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s %s" % (valtext, suffix))
