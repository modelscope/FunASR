# -*- coding: utf-8 -*-


from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_EL(lang_EU.Num2Word_EU):
    def set_high_numwords(self, high):
        max = 3 + 3 * len(high)
        for word, n in zip(high, range(max, 3, -3)):
            self.cards[10 ** n] = word # Todo: check "-illion"

    def setup(self):
        super(Num2Word_EL, self).setup()
        # lang_EU.Num2Word_EU.setup(self)

        self.negword = "μείον "
        self.pointword = "σημείο"
        self.exclude_title = ["και", "σημείο", "μείον"]

        self.high_numwords = [(1000000000, "ένα δισεκατομμύριο"),
                             (100000000, "εκατό εκατομμύρια"),
                             (10000000, "δέκα εκατομμύρια"),
                             (1000000, "ένα εκατομμύριο"),
                             (100000, "εκατό χιλιάδες"),
                             (10000, "δέκα χιλιάδες")]

        self.mid_numwords = [(1000, "χίλια"), (100, "εκατό"),
                             (90, "ενενήντα"), (80, "ογδόντα"), (70, "εβδομήντα"),
                             (60, "εξήντα"), (50, "πενήντα"), (40, "σαράντα"),
                             (30, "τριάντα")]

        self.low_numwords = ["είκοσι", 
                            "δεκαεννέα", 
                            "δεκαοχτώ", 
                            "δεκαεπτά",
                            "δεκαέξι", 
                            "δεκαπέντε", 
                            "δεκατέσσερα", 
                            "δεκατρία",
                            "δώδεκα", 
                            "έντεκα", 
                            "δέκα", 
                            "εννιά", 
                            "οχτώ",
                            "επτά", 
                            "έξι", 
                            "πέντε", 
                            "τέσσερα", 
                            "τρία", 
                            "δύο",
                            "ένα", 
                            "μηδέν"]

        self.ords = {"ένα": "πρώτος",
                     "δύο": "δεύτερος",
                     "τρία": "τρίτος",
                     "τέσσερα": "τέταρτος",
                     "πέντε": "πέμπτος",
                     "έξι": "έκτος",
                     "επτά": "έβδομος",
                     "οχτώ": "όγδοος",
                     "εννιά": "ένατος",
                     "δέκα": "δέκατος",
                     "έντεκα": "ενδέκατος",
                     "δώδεκα": "δωδέκατος"}

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
            ordinal_word = value + "τος"
        return ordinal_word

    def to_year(self, val, suffix=None, longval=True):
        if val < 0:
            val = abs(val)
            suffix = 'π.Χ' if not suffix else suffix
        high, low = (val // 100, val % 100)
        # If year is 00XX, X00X, or beyond 9999, go cardinal.
        if (high == 0
                or (high % 10 == 0 and low < 10)
                or high >= 100):
            valtext = self.to_cardinal(val)
        else:
            hightext = self.to_cardinal(high)
            if low == 0:
                lowtext = "εκατό"
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s %s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s %s" % (valtext, suffix))
