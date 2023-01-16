# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

from .lang_EU import Num2Word_EU


class Num2Word_FR(Num2Word_EU):
    CURRENCY_FORMS = {
        'EUR': (('euro', 'euros'), ('centime', 'centimes')),
        'USD': (('dollar', 'dollars'), ('cent', 'cents')),
        'FRF': (('franc', 'francs'), ('centime', 'centimes')),
        'GBP': (('livre', 'livres'), ('penny', 'pence')),
        'CNY': (('yuan', 'yuans'), ('fen', 'jiaos')),
    }

    def setup(self):
        Num2Word_EU.setup(self)

        self.negword = "moins "
        self.pointword = "virgule"
        self.errmsg_nonnum = (
            u"Seulement des nombres peuvent être convertis en mots."
            )
        self.errmsg_toobig = u"Nombre trop grand pour être converti en mots."
        self.exclude_title = ["et", "virgule", "moins"]
        self.mid_numwords = [(1000, "mille"), (100, "cent"),
                             (80, "quatre-vingts"), (60, "soixante"),
                             (50, "cinquante"), (40, "quarante"),
                             (30, "trente")]
        self.low_numwords = ["vingt", "dix-neuf", "dix-huit", "dix-sept",
                             "seize", "quinze", "quatorze", "treize", "douze",
                             "onze", "dix", "neuf", "huit", "sept", "six",
                             "cinq", "quatre", "trois", "deux", "un", "zéro"]
        self.ords = {
            "cinq": "cinquième",
            "neuf": "neuvième",
        }

    def merge(self, curr, next):
        ctext, cnum, ntext, nnum = curr + next

        if cnum == 1:
            if nnum < 1000000:
                return next
        else:
            if (not (cnum - 80) % 100
                or (not cnum % 100 and cnum < 1000))\
                    and nnum < 1000000 \
                    and ctext[-1] == "s":
                ctext = ctext[:-1]
            if cnum < 1000 and nnum != 1000 and \
                    ntext[-1] != "s" and not nnum % 100:
                ntext += "s"

        if nnum < cnum < 100:
            if nnum % 10 == 1 and cnum != 80:
                return ("%s et %s" % (ctext, ntext), cnum + nnum)
            return ("%s-%s" % (ctext, ntext), cnum + nnum)
        if nnum > cnum:
            return ("%s %s" % (ctext, ntext), cnum * nnum)
        return ("%s %s" % (ctext, ntext), cnum + nnum)

    # Is this right for such things as 1001 - "mille unième" instead of
    # "mille premier"??  "millième"??

    def to_ordinal(self, value):
        self.verify_ordinal(value)
        if value == 1:
            return "premier"
        word = self.to_cardinal(value)
        for src, repl in self.ords.items():
            if word.endswith(src):
                word = word[:-len(src)] + repl
                break
        else:
            if word[-1] == "e":
                word = word[:-1]
            word = word + "ième"
        return word

    def to_ordinal_num(self, value):
        self.verify_ordinal(value)
        out = str(value)
        out += "er" if value == 1 else "me"
        return out

    def to_currency(self, val, currency='EUR', cents=True, separator=' et',
                    adjective=False):
        result = super(Num2Word_FR, self).to_currency(
            val, currency=currency, cents=cents, separator=separator,
            adjective=adjective)
        return result
