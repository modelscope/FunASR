# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

from .lang_ES import Num2Word_ES


class Num2Word_ES_CO(Num2Word_ES):

    def to_currency(self, val, longval=True, old=False):
        result = self.to_splitnum(val, hightxt="peso/s", lowtxt="centavo/s",
                                  divisor=1, jointxt="y", longval=longval)
        # Handle exception, in spanish is "un euro" and not "uno euro"
        return result.replace("uno", "un")
