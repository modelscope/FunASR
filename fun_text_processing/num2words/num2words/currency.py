# -*- coding: utf-8 -*-

from __future__ import division

from decimal import ROUND_HALF_UP, Decimal


def parse_currency_parts(value, is_int_with_cents=True):
    if isinstance(value, int):
        if is_int_with_cents:
            # assume cents if value is integer
            negative = value < 0
            value = abs(value)
            integer, cents = divmod(value, 100)
        else:
            negative = value < 0
            integer, cents = abs(value), 0

    else:
        value = Decimal(value)
        value = value.quantize(
            Decimal('.01'),
            rounding=ROUND_HALF_UP
        )
        negative = value < 0
        value = abs(value)
        integer, fraction = divmod(value, 1)
        integer = int(integer)
        cents = int(fraction * 100)

    return integer, cents, negative


def prefix_currency(prefix, base):
    return tuple("%s %s" % (prefix, i) for i in base)
