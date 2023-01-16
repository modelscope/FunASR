# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from .base import Num2Word_Base

GENERIC_DOLLARS = ('dollar', 'dollars')
GENERIC_CENTS = ('cent', 'cents')


class Num2Word_EU(Num2Word_Base):
    CURRENCY_FORMS = {
        'AUD': (GENERIC_DOLLARS, GENERIC_CENTS),
        'CAD': (GENERIC_DOLLARS, GENERIC_CENTS),
        # repalced by EUR
        'EEK': (('kroon', 'kroons'), ('sent', 'senti')),
        'EUR': (('euro', 'euro'), GENERIC_CENTS),
        'GBP': (('pound sterling', 'pounds sterling'), ('penny', 'pence')),
        # replaced by EUR
        'LTL': (('litas', 'litas'), GENERIC_CENTS),
        # replaced by EUR
        'LVL': (('lat', 'lats'), ('santim', 'santims')),
        'USD': (GENERIC_DOLLARS, GENERIC_CENTS),
        'RUB': (('rouble', 'roubles'), ('kopek', 'kopeks')),
        'SEK': (('krona', 'kronor'), ('öre', 'öre')),
        'NOK': (('krone', 'kroner'), ('øre', 'øre')),
        'PLN': (('zloty', 'zlotys', 'zlotu'), ('grosz', 'groszy')),
        'MXN': (('peso', 'pesos'), GENERIC_CENTS),
        'RON': (('leu', 'lei', 'de lei'), ('ban', 'bani', 'de bani')),
        'INR': (('rupee', 'rupees'), ('paisa', 'paise')),
        'HUF': (('forint', 'forint'), ('fillér', 'fillér'))
    }

    CURRENCY_ADJECTIVES = {
        'AUD': 'Australian',
        'CAD': 'Canadian',
        'EEK': 'Estonian',
        'USD': 'US',
        'RUB': 'Russian',
        'NOK': 'Norwegian',
        'MXN': 'Mexican',
        'RON': 'Romanian',
        'INR': 'Indian',
        'HUF': 'Hungarian'
    }

    GIGA_SUFFIX = "illiard"
    MEGA_SUFFIX = "illion"

    def set_high_numwords(self, high):
        cap = 3 + 6 * len(high)

        for word, n in zip(high, range(cap, 3, -6)):
            if self.GIGA_SUFFIX:
                self.cards[10 ** n] = word + self.GIGA_SUFFIX

            if self.MEGA_SUFFIX:
                self.cards[10 ** (n - 3)] = word + self.MEGA_SUFFIX

    def gen_high_numwords(self, units, tens, lows):
        out = [u + t for t in tens for u in units]
        out.reverse()
        return out + lows

    def pluralize(self, n, forms):
        form = 0 if n == 1 else 1
        return forms[form]

    def setup(self):
        lows = ["non", "oct", "sept", "sext", "quint", "quadr", "tr", "b", "m"]
        units = ["", "un", "duo", "tre", "quattuor", "quin", "sex", "sept",
                 "octo", "novem"]
        tens = ["dec", "vigint", "trigint", "quadragint", "quinquagint",
                "sexagint", "septuagint", "octogint", "nonagint"]
        self.high_numwords = ["cent"] + self.gen_high_numwords(units, tens,
                                                               lows)
