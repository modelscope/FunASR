# -*- coding: utf-8 -*-

# Lao number to words

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_LO(lang_EU.Num2Word_EU):
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
        super(Num2Word_LO, self).setup()

        self.negword = "ລົບ"
        self.pointword = "ຈຸດ"
        self.exclude_title = ["ແລະ", "ຈຸດ", "ລົບ"]

        self.high_numwords = [(10000000000, 'ສິບຕື້'),
                              (1000000000, 'ຫນຶ່ງຕື້'),
                              (100000000, 'ຫນຶ່ງຮ້ອຍລ້ານ'), 
                              (10000000,'ສິບລ້ານ'), 
                              (1000000, 'ຫນຶ່ງລ້ານ'), 
                              (100000, 'ຫນຶ່ງແສນ'), 
                              (1000, 'ຫນຶ່ງພັນ') ]
        self.mid_numwords = [(1000, "ຫນຶ່ງພັນ"), (100, "ຫນຶ່ງຮ້ອຍ"),
                             (90, "ເກົ້າສິບ"), (80, "ແປດສິບ"), (70, "ເຈັດສິບ"),
                             (60, "ຫົກສິບ"), (50, "ຫ້າສິບ"), (40, "ສີ່ສິບ"),
                             (30, "ສາມສິບ")]

        self.low_numwords = ['ຊາວ', 'ສິບເກົ້າ', 'ສິບແປດ', 'ສິບເຈັດ', 'ສິບຫົກ', 'ສິບຫ້າ', 'ສິບສີ່', 'ສິບສາມ', 'ສິບສອງ', 'ສິບເອັດ', 'ສິບc', 'ເກົ້າ', 'ແປດ', 'ເຈັດ', 'ຫົກ', 'ຫ້າ', 'ສີ່', 'ສາມ', 'ສອງ', 'ຫນຶ່ງ', 'ສູນ']


        self.ords = {"ຫນຶ່ງ": "ທໍາອິດ",
                    "ສອງ": "ທີສອງ",
                    "ສາມ": "ທີສາມ",
                    "ສີ່": "ທີສີ່",
                    "ຫ້າ": "ທີຫ້າ",
                    "ຫົກ":"ທີຫົກ",
                    "ເຈັດ": "ທີເຈັດ",
                    "ແປດ": "ທີແປດ",
                    "ເກົ້າ": "ເກົ້າ",
                    "ສິບc": "ທີສິບ"
                    }


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
            suffix = 'ກ່ອນສະຕະວັດ ' if not suffix else suffix
        high, low = (val // 100, val % 100)
        # If year is 00XX, X00X, or beyond 9999, go cardinal.
        if (high == 0
                or (high % 10 == 0 and low < 10)
                or high >= 100):
            valtext = self.to_cardinal(val)
        else:
            hightext = self.to_cardinal(high)
            if low == 0:
                lowtext = "ຮ້ອຍ"
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s %s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s %s" % (valtext, suffix))
