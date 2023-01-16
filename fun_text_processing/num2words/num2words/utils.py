# -*- coding: utf-8 -*-


def splitbyx(n, x, format_int=True):
    length = len(n)
    if length > x:
        start = length % x
        if start > 0:
            result = n[:start]
            yield int(result) if format_int else result
        for i in range(start, length, x):
            result = n[i:i+x]
            yield int(result) if format_int else result
    else:
        yield int(n) if format_int else n


def get_digits(n):
    a = [int(x) for x in reversed(list(('%03d' % n)[-3:]))]
    return a
