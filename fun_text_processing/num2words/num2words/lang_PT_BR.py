# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals

import re

from . import lang_PT


class Num2Word_PT_BR(lang_PT.Num2Word_PT):
    def set_high_numwords(self, high):
        max = 3 + 3*len(high)
        for word, n in zip(high, range(max, 3, -3)):
            self.cards[10**n] = word + "ilhão"

    def setup(self):
        super(Num2Word_PT_BR, self).setup()

        self.low_numwords[1] = 'dezenove'
        self.low_numwords[3] = 'dezessete'
        self.low_numwords[4] = 'dezesseis'

        self.thousand_separators = {
            3: "milésimo",
            6: "milionésimo",
            9: "bilionésimo",
            12: "trilionésimo",
            15: "quadrilionésimo"
        }

    def merge(self, curr, next):
        ctext, cnum, ntext, nnum = curr + next

        if cnum == 1:
            if nnum < 1000000:
                return next
            ctext = "um"
        elif cnum == 100 and not nnum == 1000:
            ctext = "cento"

        if nnum < cnum:
            return ("%s e %s" % (ctext, ntext), cnum + nnum)

        elif (not nnum % 1000000) and cnum > 1:
            ntext = ntext[:-4] + "lhões"

        if nnum == 100:
            ctext = self.hundreds[cnum]
            ntext = ""

        else:
            ntext = " " + ntext

        return (ctext + ntext, cnum * nnum)

    def to_cardinal(self, value):
        result = lang_PT.Num2Word_EU.to_cardinal(self, value)

        # Transforms "mil E cento e catorze reais" into "mil, cento e catorze
        # reais"
        for ext in (
                'mil', 'milhão', 'milhões', 'bilhão', 'bilhões',
                'trilhão', 'trilhões', 'quatrilhão', 'quatrilhões'):
            if re.match('.*{} e \\w*ento'.format(ext), result):
                result = result.replace(
                    '{} e'.format(ext), '{},'.format(ext), 1
                )

        return result

    def to_currency(self, val, longval=True):
        integer_part, decimal_part = ('%.2f' % val).split('.')

        result = self.to_cardinal(int(integer_part))

        appended_currency = False
        for ext in (
                'milhão', 'milhões', 'bilhão', 'bilhões',
                'trilhão', 'trilhões', 'quatrilhão', 'quatrilhões'):
            if result.endswith(ext):
                result += ' de reais'
                appended_currency = True

        if result in ['um', 'menos um']:
            result += ' real'
            appended_currency = True
        if not appended_currency:
            result += ' reais'

        if int(decimal_part):
            cents = self.to_cardinal(int(decimal_part))
            result += ' e ' + cents

            if cents == 'um':
                result += ' centavo'
            else:
                result += ' centavos'

        return result
