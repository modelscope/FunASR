# -*- coding: utf-8 -*-

# Bulgarian num2words

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_BG(lang_EU.Num2Word_EU):
    def set_high_numwords(self, high):
        max = 3 + 3 * len(high)
        for word, n in zip(high, range(max, 3, -3)):
            self.cards[10 ** n] = word + " милиона"

    def setup(self):
        super(Num2Word_BG, self).setup()

        self.negword = "минус "
        self.pointword = "точка"
        self.exclude_title = ["и", "точка", "минус"]

        self.mid_numwords = [(1000, "хиляда"), (100, "сто"),
                             (90, "деветдесет"), (80, "осемдесет"), (70, "седемдесет"),
                             (60, "шестдесет"), (50, "петдесет"), (40, "четиридесет"),
                             (30, "тридесет")]
        self.low_numwords = ['двадесет', 'деветнайсет', 'осемнадесет', 'седемнадесет', 'шестнадесет', 'петнадесет', 'четиринадесет', 'тринадесет', 'дванадесет', 'единадесет', 'десет', 'девет', 'осем', 'седем', 'шест', 'пет', 'четири', 'три', 'две', 'един', 'нула']
        self.ords = {'един': 'първи',
'две': 'второ',
'три': 'трети',
'четири': 'четвърто',
'пет': 'пети',
'шест': 'шесто',
'седем': 'седмо',
'осем': 'осмо',
'девет': 'девети',
'десет': 'десети'}

    def merge(self, lpair, rpair):
        ltext, lnum = lpair
        rtext, rnum = rpair
        if lnum == 1 and rnum < 100:
            return (rtext, rnum)
        elif 100 > lnum > rnum:
            return ("%s-%s" % (ltext, rtext), lnum + rnum)
        elif lnum >= 100 > rnum:
            return ("%s и %s" % (ltext, rtext), lnum + rnum)
        elif rnum > lnum:
            return ("%s %s" % (ltext, rtext), lnum * rnum)
        return ("%s, %s" % (ltext, rtext), lnum + rnum)

    # def to_ordinal(self, value):
    #     self.verify_ordinal(value)
    #     outwords = self.to_cardinal(value).split(" ")
    #     lastwords = outwords[-1].split("-")
    #     lastword = lastwords[-1].lower()
    #     try:
    #         lastword = self.ords[lastword]
    #     except KeyError:
    #         if lastword[-1] == "y":
    #             lastword = lastword[:-1] + "ie"
    #         lastword += "th"
    #     lastwords[-1] = self.title(lastword)
    #     outwords[-1] = "-".join(lastwords)
    #     return " ".join(outwords)

    # def to_ordinal_num(self, value):
    #     self.verify_ordinal(value)
    #     return "%s%s" % (value, self.to_ordinal(value)[-2:])

    def to_year(self, val, suffix=None, longval=True):
        if val < 0:
            val = abs(val)
            suffix = 'г. пр.н.е' if not suffix else suffix
        high, low = (val // 100, val % 100)
        # If year is 00XX, X00X, or beyond 9999, go cardinal.
        if (high == 0
                or (high % 10 == 0 and low < 10)
                or high >= 100):
            valtext = self.to_cardinal(val)
        else:
            hightext = self.to_cardinal(high)
            if low == 0:
                lowtext = "сто"
            elif low < 10:
                lowtext = "oh-%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s %s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s %s" % (valtext, suffix))
