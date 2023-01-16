# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from .lang_EN import Num2Word_EN


class Num2Word_EN_IN(Num2Word_EN):
    def set_high_numwords(self, high):
        self.cards[10 ** 7] = "crore"
        self.cards[10 ** 5] = "lakh"
