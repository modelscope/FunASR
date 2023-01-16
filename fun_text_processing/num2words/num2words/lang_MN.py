# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_MN(lang_EU.Num2Word_EU):
    # def set_high_numwords(self, high):
    #     max = 3 + 3 * len(high)
    #     for word, n in zip(high, range(max, 3, -3)):
    #         self.cards[10 ** n] = word
    def set_high_numwords(self, high):
        max = 3 * len(high)
        for word, n in zip(high, range(max, 3, -1)):
            # print(word[0],word[1],n)
            self.cards[10 ** n] = word[1]
            # self.cards[10**n] = word + "លាន"
        # try:
        #     ordinal_word = self.high_numwords[high]
        # except KeyError:
        #     max = 3 + 3 * len(high)
        #     for word, n in zip(high, range(max, 3, -3)):
        #         print(word)
        #         print(n)
        #         self.cards[10 ** n] = word

    def setup(self):
        super(Num2Word_MN, self).setup()

        self.negword = 'хасах'
        self.pointword = 'цэг'

        self.high_numwords = [(1000000,'нэг сая'), (100000, 'нэг зуун мянга'), (10000, 'арван мянга'), (1000, 'нэг мянга') ]

        self.mid_numwords = [(100,'нэг зуу'), (90,'ерэн'), (80,'наян'), (70,'далан'), (60,'жаран'), (50,'тавин'), (40,'дөчин'), (30,'гучин')]

        self.low_numwords = ['хорин','арван есөн','арван найман','арван долоон','арван зургаа','арван тав','арван дөрөв','арван гурав','арван хоёр','арван нэгэн','арав','есөн','найм','Долоо','зургаа','тав','дөрөв','гурав','хоёр','нэг','тэг']

        self.ords = {'нэг': 'эхлээд',
                    'хоёр': 'хоёрдугаарт',
                    'гурав': 'гурав дахь',
                    'дөрөв': 'урагш',
                    'тав': 'тав дахь',
                    'зургаа': 'зургаа дахь',
                    'Долоо': 'долоо дахь',
                    'найм': 'найм дахь',
                    'есөн': 'ес дэх',
                    'арав': 'Аравдугаар'}


    def merge(self, lpair, rpair):
        ltext, lnum = lpair
        rtext, rnum = rpair
        if lnum == 1 and rnum < 100:
            return (rtext, rnum)
        elif 100 > lnum > rnum:
            return ("%s %s" % (ltext, rtext), lnum + rnum)
        elif lnum >= 100 > rnum:
            return ("%s болон %s" % (ltext, rtext), lnum + rnum)
        elif rnum > lnum:
            return ("%s %s" % (ltext, rtext), lnum * rnum)
        return ("%s %s" % (ltext, rtext), lnum + rnum)

    def to_ordinal_num(self, value):
        self.verify_ordinal(value)
        return "%s%s" % (value, self.to_ordinal(value))

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
                lowtext = "зуун"
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s%s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s%s" % (valtext, suffix))
