# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from .base import Num2Word_Base
from .utils import get_digits, splitbyx

ZERO = ('ноль',)

ONES_FEMININE = {
    1: ('одна',),
    2: ('две',),
    3: ('три',),
    4: ('четыре',),
    5: ('пять',),
    6: ('шесть',),
    7: ('семь',),
    8: ('восемь',),
    9: ('девять',),
}

ONES = {
    1: ('один',),
    2: ('два',),
    3: ('три',),
    4: ('четыре',),
    5: ('пять',),
    6: ('шесть',),
    7: ('семь',),
    8: ('восемь',),
    9: ('девять',),
}

TENS = {
    0: ('десять',),
    1: ('одиннадцать',),
    2: ('двенадцать',),
    3: ('тринадцать',),
    4: ('четырнадцать',),
    5: ('пятнадцать',),
    6: ('шестнадцать',),
    7: ('семнадцать',),
    8: ('восемнадцать',),
    9: ('девятнадцать',),
}

TWENTIES = {
    2: ('двадцать',),
    3: ('тридцать',),
    4: ('сорок',),
    5: ('пятьдесят',),
    6: ('шестьдесят',),
    7: ('семьдесят',),
    8: ('восемьдесят',),
    9: ('девяносто',),
}

HUNDREDS = {
    1: ('сто',),
    2: ('двести',),
    3: ('триста',),
    4: ('четыреста',),
    5: ('пятьсот',),
    6: ('шестьсот',),
    7: ('семьсот',),
    8: ('восемьсот',),
    9: ('девятьсот',),
}

THOUSANDS = {
    1: ('тысяча', 'тысячи', 'тысяч'),  # 10^3
    2: ('миллион', 'миллиона', 'миллионов'),  # 10^6
    3: ('миллиард', 'миллиарда', 'миллиардов'),  # 10^9
    4: ('триллион', 'триллиона', 'триллионов'),  # 10^12
    5: ('квадриллион', 'квадриллиона', 'квадриллионов'),  # 10^15
    6: ('квинтиллион', 'квинтиллиона', 'квинтиллионов'),  # 10^18
    7: ('секстиллион', 'секстиллиона', 'секстиллионов'),  # 10^21
    8: ('септиллион', 'септиллиона', 'септиллионов'),  # 10^24
    9: ('октиллион', 'октиллиона', 'октиллионов'),  # 10^27
    10: ('нониллион', 'нониллиона', 'нониллионов'),  # 10^30
}


class Num2Word_RU(Num2Word_Base):
    CURRENCY_FORMS = {
        'RUB': (
            ('рубль', 'рубля', 'рублей'), ('копейка', 'копейки', 'копеек')
        ),
        'EUR': (
            ('евро', 'евро', 'евро'), ('цент', 'цента', 'центов')
        ),
        'USD': (
            ('доллар', 'доллара', 'долларов'), ('цент', 'цента', 'центов')
        ),
        'UAH': (
            ('гривна', 'гривны', 'гривен'), ('копейка', 'копейки', 'копеек')
        ),
        'KZT': (
            ('тенге', 'тенге', 'тенге'), ('тиын', 'тиына', 'тиынов')
        ),
    }

    def setup(self):
        self.negword = "минус"
        self.pointword = "запятая"
        self.ords = {"ноль": "нулевой",
                     "один": "первый",
                     "два": "второй",
                     "три": "третий",
                     "четыре": "четвертый",
                     "пять": "пятый",
                     "шесть": "шестой",
                     "семь": "седьмой",
                     "восемь": "восьмой",
                     "девять": "девятый",
                     "сто": "сотый"}
        self.ords_feminine = {"один": "",
                              "одна": "",
                              "две": "двух",
                              "три": "трёх",
                              "четыре": "четырёх",
                              "пять": "пяти",
                              "шесть": "шести",
                              "семь": "семи",
                              "восемь": "восьми",
                              "девять": "девяти"}

    def to_cardinal(self, number):
        n = str(number).replace(',', '.')
        if '.' in n:
            left, right = n.split('.')
            leading_zero_count = len(right) - len(right.lstrip('0'))
            decimal_part = ((ZERO[0] + ' ') * leading_zero_count +
                            self._int2word(int(right)))
            return u'%s %s %s' % (
                self._int2word(int(left)),
                self.pointword,
                decimal_part
            )
        else:
            return self._int2word(int(n))

    def pluralize(self, n, forms):
        if n % 100 < 10 or n % 100 > 20:
            if n % 10 == 1:
                form = 0
            elif 5 > n % 10 > 1:
                form = 1
            else:
                form = 2
        else:
            form = 2
        return forms[form]

    def to_ordinal(self, number):
        self.verify_ordinal(number)
        outwords = self.to_cardinal(number).split(" ")
        lastword = outwords[-1].lower()
        try:
            if len(outwords) > 1:
                if outwords[-2] in self.ords_feminine:
                    outwords[-2] = self.ords_feminine.get(
                        outwords[-2], outwords[-2])
                elif outwords[-2] == 'десять':
                    outwords[-2] = outwords[-2][:-1] + 'и'
            if len(outwords) == 3:
                if outwords[-3] in ['один', 'одна']:
                    outwords[-3] = ''
            lastword = self.ords[lastword]
        except KeyError:
            if lastword[:-3] in self.ords_feminine:
                lastword = self.ords_feminine.get(
                    lastword[:-3], lastword) + "сотый"
            elif lastword[-1] == "ь" or lastword[-2] == "т":
                lastword = lastword[:-1] + "ый"
            elif lastword[-1] == "к":
                lastword = lastword + "овой"
            elif lastword[-5:] == "десят":
                lastword = lastword.replace('ь', 'и') + 'ый'
            elif lastword[-2] == "ч" or lastword[-1] == "ч":
                if lastword[-2] == "ч":
                    lastword = lastword[:-1] + "ный"
                if lastword[-1] == "ч":
                    lastword = lastword + "ный"
            elif lastword[-1] == "н" or lastword[-2] == "н":
                lastword = lastword[:lastword.rfind('н') + 1] + "ный"
            elif lastword[-1] == "д" or lastword[-2] == "д":
                lastword = lastword[:lastword.rfind('д') + 1] + "ный"
        outwords[-1] = self.title(lastword)
        return " ".join(outwords).strip()

    def _money_verbose(self, number, currency):
        return self._int2word(number, currency == 'UAH')

    def _cents_verbose(self, number, currency):
        return self._int2word(number, currency in ('UAH', 'RUB'))

    def _int2word(self, n, feminine=False):
        if n < 0:
            return ' '.join([self.negword, self._int2word(abs(n))])

        if n == 0:
            return ZERO[0]

        words = []
        chunks = list(splitbyx(str(n), 3))
        i = len(chunks)
        for x in chunks:
            i -= 1

            if x == 0:
                continue

            n1, n2, n3 = get_digits(x)

            if n3 > 0:
                words.append(HUNDREDS[n3][0])

            if n2 > 1:
                words.append(TWENTIES[n2][0])

            if n2 == 1:
                words.append(TENS[n1][0])
            elif n1 > 0:
                ones = ONES_FEMININE if i == 1 or feminine and i == 0 else ONES
                words.append(ones[n1][0])

            if i > 0:
                words.append(self.pluralize(x, THOUSANDS[i]))

        return ' '.join(words)
