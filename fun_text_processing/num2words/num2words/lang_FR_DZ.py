# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from .lang_FR import Num2Word_FR


class Num2Word_FR_DZ(Num2Word_FR):
    CURRENCY_FORMS = {
        'DIN': (('dinard', 'dinards'), ('centime', 'centimes')),
    }

    def to_currency(self, val, currency='DIN', cents=True, separator=' et',
                    adjective=False):
        result = super(Num2Word_FR, self).to_currency(
            val, currency=currency, cents=cents, separator=separator,
            adjective=adjective)
        return result
