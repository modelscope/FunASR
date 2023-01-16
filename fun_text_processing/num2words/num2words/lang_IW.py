# -*- coding: utf-8 -*-

# Hebrew num2words

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_IW(lang_EU.Num2Word_EU):
    def set_high_numwords(self, high):
        max = 3 + 3 * len(high)
        for word, n in zip(high, range(max, 3, -3)):
            self.cards[10 ** n] = word + " מִילִיוֹן"

    def setup(self):
        super(Num2Word_IW, self).setup()

        self.negword = "פחות "
        self.pointword = "נְקוּדָה"
        self.exclude_title = ["ו", "נְקוּדָה", "פחות"]

        self.mid_numwords = [(1000, "אלף"), (100, "מאה"),
                             (90, "תִשׁעִים"), (80, "שמונים"), (70, "שִׁבעִים"),
                             (60, "שִׁשִׁים"), (50, "חמישים"), (40, "ארבעים"),
                             (30, "שְׁלוֹשִׁים")]
        self.low_numwords = ['עשרים', 'תשע עשרה', 'שמונה עשרה', 'שבע עשרה', 'שש עשרה', 'חֲמֵשׁ עֶשׂרֵה', 'ארבעה עשר', 'שְׁלוֹשׁ עֶשׂרֵה', 'שתיים עשרה', 'אחד עשר', 'עשר', 'תֵשַׁע', 'שמונה', 'שבע', 'שֵׁשׁ', 'חָמֵשׁ', 'ארבע', 'שְׁלוֹשָׁה', 'שתיים', 'אחד', 'אֶפֶס']
        self.ords = {'אחד': 'ראשון',
'שתיים': 'שְׁנִיָה',
'שְׁלוֹשָׁה': 'שְׁלִישִׁי',
'ארבע': 'רביעי',
'חָמֵשׁ': 'חמישי',
'שֵׁשׁ': 'שִׁשִׁית',
'שבע': 'שְׁבִיעִית',
'שמונה': 'שמונה',
'תֵשַׁע': 'ט',
'עשר': 'עֲשִׂירִית'}

    def merge(self, lpair, rpair):
        ltext, lnum = lpair
        rtext, rnum = rpair
        if lnum == 1 and rnum < 100:
            return (rtext, rnum)
        elif 100 > lnum > rnum:
            return ("%s-%s" % (ltext, rtext), lnum + rnum)
        elif lnum >= 100 > rnum:
            return ("%s ו %s" % (ltext, rtext), lnum + rnum)
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
            suffix = 'לִפנֵי הַסְפִירָה' if not suffix else suffix
        high, low = (val // 100, val % 100)
        # If year is 00XX, X00X, or beyond 9999, go cardinal.
        if (high == 0
                or (high % 10 == 0 and low < 10)
                or high >= 100):
            valtext = self.to_cardinal(val)
        else:
            hightext = self.to_cardinal(high)
            if low == 0:
                lowtext = "מֵאָה"
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s %s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s %s" % (valtext, suffix))
