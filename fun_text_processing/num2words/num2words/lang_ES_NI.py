# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

from .lang_ES import Num2Word_ES


class Num2Word_ES_NI(Num2Word_ES):
    CURRENCY_FORMS = {
        'NIO': (('córdoba', 'córdobas'), ('centavo', 'centavos')),
    }

    def to_currency(self, val, currency='NIO', cents=True, separator=' con',
                    adjective=False):
        result = super(Num2Word_ES, self).to_currency(
            val, currency=currency, cents=cents, separator=separator,
            adjective=adjective)
        return result.replace("uno", "un")
