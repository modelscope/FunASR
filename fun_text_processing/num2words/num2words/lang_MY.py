# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_MY(lang_EU.Num2Word_EU):
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
        super(Num2Word_MY, self).setup()

        self.negword = 'အမှတ်'
        self.pointword = 'အမှတ်'

        self.high_numwords = [(1000000,'သန်း'), (100000, 'သိန်း'), (10000, 'သောင်း'), (1000, 'ထောင်') ]

        self.mid_numwords = [(100,'ရာ'), (90,'ကိုးဆယ်'), (80,'ရှစ်ဆယ်'), (70,'ခုနှစ်ဆယ်'), (60,'ခြောက်ဆယ့်'), (50,'ငါးဆယ်'), (40,'လေးဆယ်'), (30,'သုံးဆယ်')]

        # self.low_numwords = ['शून्य', 'एक', 'दुई', 'तीन', 'चार', 'पाँच', 'छ', 'सात', 'आठ', 'नौ', 'दस', 'एघार', 'बाह्र', 'तेह्र', 'चौध', 'पन्ध्र', 'सोह्र', 'सत्रह', 'अठार', 'उन्नीस']
        self.low_numwords = ['နှစ်ဆယ်','ဆယ့်ကိုး','ဆယ့်ရှစ်','ဆယ့်ခုနှစ်','ဆယ့်ခြောက်','ဆယ့်ငါး','ဆယ့်လေး','ဆယ့်သုံး','ဆယ့်နှစ်','ဆယ့်တစ်','ဆယ်','ကိုး','ရှစ်','ခုနှစ်','ခြောက်','ငါး','လေး','သုံး','နှစ်','တစ်','သုည']

        self.ords = {'တစ်':'ပထမ',
                    'နှစ်':'ဒုတိယ',
                    'သုံး':'တတိယ',
                    'လေး':'စတုတ္ထ',
                    'ငါး':'ပဉ္စမ',
                    'ခြောက်':'ဆဋ္ဌမ',
                    'ခုနှစ်':'သတ္တမ',
                    'ရှစ်':'အဋ္ဌမ',
                    'ကိုး':'နဝမ',
                    'ဆယ်':'ဒသမ'}
    def merge(self, lpair, rpair):
        ltext, lnum = lpair
        rtext, rnum = rpair
        if lnum == 1 and rnum < 100:
            return (rtext, rnum)
        elif 100 > lnum > rnum:
            return ("%s %s" % (ltext, rtext), lnum + rnum)
        elif lnum >= 100 > rnum:
            return ("%s နှင့် %s" % (ltext, rtext), lnum + rnum)
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
                lowtext = "ရာ့"
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s%s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s%s" % (valtext, suffix))
