# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

from . import lang_EU

# Bengali

class Num2Word_BN(lang_EU.Num2Word_EU):
    def set_high_numwords(self, high):
        max = 3 + 3 * len(high)
        for word, n in zip(high, range(max, 3, -3)):
            self.cards[10 ** n] = word

    def setup(self):
        super(Num2Word_BN, self).setup()

        self.negword = "বিয়োগ "
        self.pointword = "পয়েন্ট"
        self.exclude_title = ["এবং", "পয়েন্ট", "বিয়োগ"]

        self.high_numwords = [(100000000000, "ট্রিলিয়ন"),
                              (1000000000, "এক ট্রিলিয়ন"),
                              (100000000, "বিলিয়ন"),
                              (10000000, "এক শত মিলিয়ন"),
                              (1000000, "কোটি"),
                              (100000, "দশ লক্ষ"),
                              (10000, "দশ হাজার")]

        self.mid_numwords = [(1000, "হাজার"),
                             (100, "শত"),
                             (90, "নব্বই"), (80, "আশি"), (70, "সত্তর"),
                             (60, "ষাট"), (50, "পঞ্চাশ"), (40, "চল্লিশ"),
                             (30, "ত্রিশ")]
        self.low_numwords = ["বিশ","উনিশ","আঠার","সতের","ষোল","পনের","চৌদ্দ","তেরো","বারো","এগারো","দশ","নয়টি","আট","সাত","ছয়","পাঁচ","চার","তিন","দুই","এক","শূন্য"]
        self.ords = {"এক": "প্রথম",
                "দুই": "দ্বিতীয়",
                "তিন": "তৃতীয়",
                "চার": "চতুর্থ",
                "পাঁচ": "পঞ্চম",
                "ছয়": "ষষ্ঠ",
                "সাত": "সপ্তম",
                "আট": "অষ্টম",
                "নয়টি": "নবম",
                "দশ": "দশম",
                "এগারো": "একাদশ",
                "বারো": "দ্বাদশ",
                "তেরো": "ত্রয়োদশ",
                "চৌদ্দ": "চতুর্দশ",
                "পনের": "পঞ্চদশ",
                "ষোল": "ষোড়শ",
                "সতের": "সপ্তদশতম",
                "আঠার": "অষ্টাদশ",
                "উনিশ": "উনিশতম",
                "বিশ": "বিংশতম"}


    def merge(self, lpair, rpair):
        ltext, lnum = lpair
        rtext, rnum = rpair
        if lnum == 1 and rnum < 100:
            return (rtext, rnum)
        elif 100 > lnum > rnum:
            return ("%s%s" % (ltext, rtext), lnum + rnum)
        elif lnum >= 100 > rnum:
            return ("%s %s" % (ltext, rtext), lnum + rnum)
        elif rnum > lnum:
            return ("%s %s" % (ltext, rtext), lnum * rnum)
        return ("%s %s" % (ltext, rtext), lnum + rnum)

    def to_ordinal(self, value):
        self.verify_ordinal(value)
        outwords = self.to_cardinal(value).split(" ")
        lastwords = outwords[-1].split("-")
        lastword = lastwords[-1].lower()
        try:
            lastword = self.ords[lastword]
        except KeyError:
            lastword += "তম"
        lastwords[-1] = self.title(lastword)
        outwords[-1] = " ".join(lastwords)
        return " ".join(outwords)

    def to_ordinal_num(self, value):
        self.verify_ordinal(value)
        return "%s%s" % (value, self.to_ordinal(value)[-2:])

    def to_year(self, val, suffix=None, longval=True):
        if val < 0:
            val = abs(val)
            suffix = 'খ্রিস্টপূর্ব' if not suffix else suffix
        high, low = (val // 100, val % 100)
        # If year is 00XX, X00X, or beyond 9999, go cardinal.
        if (high == 0
                or (high % 10 == 0 and low < 10)
                or high >= 100):
            valtext = self.to_cardinal(val)
        else:
            hightext = self.to_cardinal(high)
            if low == 0:
                lowtext = "শত"
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s %s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s %s" % (valtext, suffix))
