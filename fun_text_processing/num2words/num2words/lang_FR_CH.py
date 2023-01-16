# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals

from .lang_FR import Num2Word_FR


class Num2Word_FR_CH(Num2Word_FR):
    def setup(self):
        Num2Word_FR.setup(self)

        self.mid_numwords = [(1000, "mille"), (100, "cent"), (90, "nonante"),
                             (80, "huitante"), (70, "septante"),
                             (60, "soixante"), (50, "cinquante"),
                             (40, "quarante"), (30, "trente")]

    def merge(self, curr, next):
        ctext, cnum, ntext, nnum = curr + next

        if cnum == 1:
            if nnum < 1000000:
                return next

        if cnum < 1000 and nnum != 1000 and\
                ntext[-1] != "s" and not nnum % 100:
            ntext += "s"

        if nnum < cnum < 100:
            if nnum % 10 == 1:
                return ("%s et %s" % (ctext, ntext), cnum + nnum)
            return ("%s-%s" % (ctext, ntext), cnum + nnum)
        if nnum > cnum:
            return ("%s %s" % (ctext, ntext), cnum * nnum)
        return ("%s %s" % (ctext, ntext), cnum + nnum)
