# -*- coding: utf-8 -*-
# Khmer

from __future__ import division, print_function, unicode_literals

from . import lang_EU

# Khmer

class Num2Word_KH(lang_EU.Num2Word_EU):
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

    def gen_high_numwords(self, units, tens, lows):
        out = [u + t for t in tens for u in units]
        out.reverse()
        return out + lows

    def setup(self):
        super(Num2Word_KH, self).setup()

        self.negword = "ដក "
        self.pointword = "ចំណុច"
        self.exclude_title = ["និង", "ចំណុច", "ដក"]
        lows = [""]
        units = [""]
        tens = ["ពាន់លាន","រយលាន","កោដិ","លាន","សែន","ម៉ឺន"]
        self.high_numwords = self.gen_high_numwords(units, tens, lows)

        self.high_numwords = [(1000000000, "មួយពាន់លាន"), 
                              (100000000, "មួយរយលាន"),
                              (10000000, "មួយកោដិ"), 
                              (1000000, "មួយលាន"), 
                              (100000, "មួយសែន"),
                              (10000, "មួយម៉ឺន")]
                             # (2400000, "ពីរលានបួនម៉ឺន")

        self.mid_numwords = [(1000, "មួយពាន់"), (100, "មួយរយ"),
                             (90, "កៅសិប"), (80, "ប៉ែតសិប"), (70, "ចិតសិប"),
                             (60, "ហុកសិប"), (50, "ហាសិប"), (40, "សែសិប"),
                             (30, "សាមសិប")]

        self.low_numwords = ["ម្ភៃ", 
                            "ដប់ប្រាំបួន", 
                            "ដប់ប្រាំបី", 
                            "ដប់ប្រាំពីរ",
                            "ដប់ប្រាំមួយ", 
                            "ដប់ប្រាំ", 
                            "ដប់បួន", 
                            "ដប់បី",
                            "ដប់ពីរ", 
                            "ដប់មួយ", 
                            "ដប់", 
                            "ប្រាំបួន", 
                            "ប្រាំបី",
                            "ប្រាំពីរ", 
                            "ប្រាំមួយ", 
                            "ប្រាំ", 
                            "បួន", 
                            "បី", 
                            "ពីរ",
                            "មួយ", 
                            "សូន្យ"]

        self.ords = {"មួយ": "ទីមួយ",
                     "ពីរ": "ទីពីរ",
                     "បី": "ទីបី",
                     "បួន": "ទីបួន",
                     "ប្រាំ": "ទីប្រាំ",
                     "ប្រាំមួយ": "ទីប្រាំមួយ",
                     "ប្រាំពីរ": "ទីប្រាំពីរ",
                     "ប្រាំបី": "ទីប្រាំបី",
                     "ប្រាំបួន": "ទីប្រាំបួន",
                     "ដប់": "ទីដប់",
                     "ដប់មួយ": "ទីដប់មួយ",
                     "ដប់ពីរ": "ទីដប់ពីរ"}

    def merge(self, lpair, rpair):
        ltext, lnum = lpair
        rtext, rnum = rpair
        if lnum == 1 and rnum < 100:
            return (rtext, rnum)
        elif 100 > lnum > rnum:
            return ("%s%s" % (ltext, rtext), lnum + rnum)
        elif lnum >= 100 > rnum:
            return ("%s%s" % (ltext, rtext), lnum + rnum)
        elif rnum > lnum:
            return ("%s%s" % (ltext, rtext), lnum * rnum)
        return ("%s%s" % (ltext, rtext), lnum + rnum)

    def to_ordinal_num(self, value):
        self.verify_ordinal(value)
        return "%s%s" % (value, self.to_ordinal(value))

    def to_ordinal(self, value):
        self.verify_ordinal(value)
        try:
            ordinal_word = self.ords[value]
        except KeyError:
            ordinal_word = "ទី" + value
        return ordinal_word


    def to_year(self, val, suffix=None, longval=True):
        if val < 0:
            val = abs(val)
            suffix = 'មុនគ' if not suffix else suffix
        high, low = (val // 100, val % 100)
        # If year is 00XX, X00X, or beyond 9999, go cardinal.
        if (high == 0
                or (high % 10 == 0 and low < 10)
                or high >= 100):
            valtext = self.to_cardinal(val)
        else:
            hightext = self.to_cardinal(high)
            if low == 0:
                lowtext = "រយ"
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s %s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s %s" % (valtext, suffix))
