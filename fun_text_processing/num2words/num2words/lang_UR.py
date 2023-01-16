# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_UR(lang_EU.Num2Word_EU):
    # def set_high_numwords(self, high):
    #     max = 3 + 3 * len(high)
    #     for word, n in zip(high, range(max, 3, -3)):
    #         self.cards[10 ** n] = word
    def reverse_text(text):
        return ''.join(reversed(text))

    def set_high_numwords(self, high):
        max = 3 * len(high)
        for word, n in zip(high, range(max, 3, -3)):
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
        super(Num2Word_UR, self).setup()

        self.negword = 'مائنس'
        self.pointword = 'پوائنٹ'

        self.high_numwords = [(1000000000000,'ٹریلین'), (1000000000, 'ایک ارب'), (1000000, 'دس لاکھ') ]

        self.mid_numwords = [(1000, 'ہزار'), (100,'سو'),(90, 'نوے'), (80, 'اسی'), (70, 'ستر'), (60, 'ساٹھ'), (50, 'پچاس'), (40, 'چالیس'), (30, 'تیس')]

        self.low_numwords = ["بیس",
                            "انیس",
                            "اٹھارہ",
                            "سترہ",
                            "سولہ",
                            "پندرہ",
                            "چودہ",
                            "تیرہ",
                            "بارہ",
                            "گیارہ",
                            "دس",
                            "نو",
                            "آٹھ",
                            "سات",
                            "چھ",
                            "پانچ",
                            "چار",
                            "تین",
                            "دو",
                            "ایک",
                            "صفر"]

        self.ords = {'ایک':'پہلا',
                    'دو':'دوسرا',
                    'تین':'تیسرے',
                    'چار':'چوتھا',
                    'پانچ':'پانچویں',
                    'چھ':'چھٹا',
                    'سات':'ساتویں',
                    'آٹھ':'آٹھویں',
                    'نو':'نویں',
                    'دس':'دسویں'}

        self.labeled_numbers = {'0':'صفر',
                                '1':'ایک',
                                '2':'دو',
                                '3':'تین',
                                '4':'چار',
                                '5':'پانچ',
                                '6':'چھ',
                                '7':'سات',
                                '8':'آٹھ',
                                '9':'نو',
                                '10':'دس',
                                '11':'گیارہ',
                                '12':'بارہ',
                                '13':'تیرہ',
                                '14':'چودہ',
                                '15':'پندرہ',
                                '16':'سولہ',
                                '17':'سترہ',
                                '18':'اٹھارہ',
                                '19':'انیس',
                                '20':'بیس',
                                '21':'اکیس',
                                '22':'بائیس',
                                '23':'تئیس',
                                '24':'چوبیس',
                                '25':'پچیس',
                                '26':'چھببیس',
                                '27':'ستائیس',
                                '28':'اٹھائیس',
                                '29':'انتیس',
                                '30':'تیس',
                                '31':'اکتیس',
                                '32':'بتیس',
                                '33':'تینتیس',
                                '34':'چونتیس',
                                '35':'پینتیس',
                                '36':'چھتیس',
                                '37':'سینتیس',
                                '38':'اڑتیس',
                                '39':'انتالیس',
                                '40':'چالیس',
                                '41':'اکتالیس',
                                '42':'بیالیس',
                                '43':'تینتالیس',
                                '44':'چوالیس',
                                '45':'پینتالیس',
                                '46':'چھیالیس',
                                '47':'سینتالیس',
                                '48':'اڑتالیس',
                                '49':'انچاس',
                                '50':'پچاس',
                                '51':'اکیاون',
                                '52':'باون',
                                '53':'ترپن',
                                '54':'چون',
                                '55':'پچپن',
                                '56':'چھپن',
                                '57':'ستاون',
                                '58':'اٹھاون',
                                '59':'انسٹھ',
                                '60':'ساٹھ',
                                '61':'اکسٹھ',
                                '62':'باسٹھ',
                                '63':'ترسٹھ',
                                '64':'چوسٹھ',
                                '65':'پینسٹھ',
                                '66':'چھیاسٹھ',
                                '67':'سڑسٹھ',
                                '68':'اٹھسٹھ',
                                '69':'انہتر',
                                '70':'ستر',
                                '71':'اکہتر',
                                '72':'بہتر',
                                '73':'تہتر',
                                '74':'چوہتر',
                                '75':'پچہتر',
                                '76':'چھہتر',
                                '77':'ستتر',
                                '78':'اٹھہتر',
                                '79':'اناسی',
                                '80':'اسی',
                                '81':'اکیاسی',
                                '82':'بیاسی',
                                '83':'تراسی',
                                '84':'چوراسی',
                                '85':'پچاسی',
                                '86':'چھیاسی',
                                '87':'ستاسی',
                                '88':'اٹھاسی',
                                '89':'نواسی',
                                '90':'نوے',
                                '91':'اکانوے',
                                '92':'بانوے',
                                '93':'ترانوے',
                                '94':'چورانوے',
                                '95':'پچانوے',
                                '96':'چھیانوے',
                                '97':'ستانوے',
                                '98':'اٹھانوے',
                                '99':'ننانوے',
                                '100':'سو',
                                '200':'دو سو',
                                '1000':'ایک ہزار',
                                '2000':'دو ہزار',
                                '100000':'ایک لاکھ',
                                '1000000':'دس لاکھ',
                                '2000000':'بیس لاکھ',
                                '10000000':'ایک کروڑ',
                                '100000000':'دس کروڑ',
                                '1000000000':'ایک ارب',
                                '10000000000':'دس ارب'}

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
                lowtext = "सय"
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s%s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s%s" % (valtext, suffix))
