# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

from . import lang_EU


class Num2Word_NE(lang_EU.Num2Word_EU):
    # def set_high_numwords(self, high):
    #     max = 3 + 3 * len(high)
    #     for word, n in zip(high, range(max, 3, -3)):
    #         self.cards[10 ** n] = word
    def set_high_numwords(self, high):
        max = 3 * len(high)
        for word, n in zip(high, range(max, 0, -3)):
            # print(word[0],word[1],n)
            self.cards[10 ** n] = word[1]
            # self.cards[10**n] = word + "លាន"
        # try:
        #     ordinal_word = self.high_numwords[high]
        # except KeyError:
        #     max = 3 + 3 * len(high)
        #     for word, n in zip(high, range(max, 3, -3)):
        #         print(word)
        #         print(n)
        #         self.cards[10 ** n] = word

    def setup(self):
        super(Num2Word_NE, self).setup()

        self.negword = 'माइनस'
        self.pointword = 'अंक'

        self.high_numwords = [(1000000000000000, 'सय ट्रिलियन'), (1000000000000,'ट्रिलियन'), (1000000000, 'बिलियन'), (1000000, 'मिलियन'), (1000, 'हजार') ]

        self.mid_numwords = [(100, 'एक सय'), (90, 'नब्बे'), (80, 'अस्सी'), (70, 'सत्तरी'), (60, 'साठ'), (50, 'पचास'), (40, 'चालीस'), (30, 'तीस'), (20, 'विस')]

        # self.low_numwords = ['शून्य', 'एक', 'दुई', 'तीन', 'चार', 'पाँच', 'छ', 'सात', 'आठ', 'नौ', 'दस', 'एघार', 'बाह्र', 'तेह्र', 'चौध', 'पन्ध्र', 'सोह्र', 'सत्रह', 'अठार', 'उन्नीस']
        self.low_numwords = ["उन्नाइस",
                             "अठार",
                             "सत्र",
                             "सोह्र",
                             "पन्ध्र",
                             "चौध",
                             "तेह्र",
                             "बाह्र",
                             "एघार",
                             "दश",
                             "नौ",
                             "आठ",
                             "सात",
                             "छ",
                             "पाँच",
                             "चार",
                             "तीन",
                             "दुई",
                             "एक",
                             "शून्य"]

        self.ords = {'एक': 'पहिलो',
                    'दुई': 'द्दोस्रो',
                    'तीन': 'तेस्रो',
                    'चार': 'चौथुर्त',
                    'पाँच': 'पाँचौ',
                    'छ': 'छैटौ',
                    'सात': 'सातौ',
                    'आठ': 'आठौ',
                    'नौ': 'नवौ',
                    'दश': 'दसौ'}

        self.labeled_numbers = {'0': 'शुन्य',
                                '1': 'एक',
                                '2': 'दुई',
                                '3': 'तीन',
                                '4': 'चार',
                                '5': 'पाँच',
                                '6': 'छ',
                                '7': 'सात',
                                '8': 'आठ',
                                '9': 'नौ',
                                '10': 'दश',
                                '11': 'एघार',
                                '12': 'बाह्र',
                                '13': 'तेह्र',
                                '14': 'चौध',
                                '15': 'पन्ध्र',
                                '16': 'सोह्र',
                                '17': 'सत्र',
                                '18': 'अठार',
                                '19': 'उन्नाइस',
                                '20': 'विस',
                                '21': 'एक्काइस',
                                '22': 'बाइस',
                                '23': 'तेईस',
                                '24': 'चौविस',
                                '25': 'पच्चिस',
                                '26': 'छब्बिस',
                                '27': 'सत्ताइस',
                                '28': 'अठ्ठाईस',
                                '29': 'उनन्तिस',
                                '30': 'तिस',
                                '31': 'एकत्तिस',
                                '32': 'बत्तिस',
                                '33': 'तेत्तिस',
                                '34': 'चौँतिस',
                                '35': 'पैँतिस',
                                '36': 'छत्तिस',
                                '37': 'सैँतीस',
                                '38': 'अठतीस',
                                '39': 'उनन्चालीस',
                                '40': 'चालीस',
                                '41': 'एकचालीस',
                                '42': 'बयालीस',
                                '43': 'त्रियालीस',
                                '44': 'चवालीस',
                                '45': 'पैँतालीस',
                                '46': 'छयालीस',
                                '47': 'सच्चालीस',
                                '48': 'अठचालीस',
                                '49': 'उनन्चास',
                                '50': 'पचास',
                                '51': 'एकाउन्न',
                                '52': 'बाउन्न',
                                '53': 'त्रिपन्न',
                                '54': 'चउन्न',
                                '55': 'पचपन्न',
                                '56': 'छपन्न',
                                '57': 'सन्ताउन्न',
                                '58': 'अन्ठाउन्न',
                                '59': 'उनन्साठी',
                                '60': 'साठी',
                                '61': 'एकसट्ठी',
                                '62': 'बयसट्ठी',
                                '63': 'त्रिसट्ठी',
                                '64': 'चौंसट्ठी',
                                '65': 'पैंसट्ठी',
                                '66': 'छयसट्ठी',
                                '67': 'सतसट्ठी',
                                '68': 'अठसट्ठी',
                                '69': 'उनन्सत्तरी',
                                '70': 'सत्तरी',
                                '71': 'एकहत्तर',
                                '72': 'बहत्तर',
                                '73': 'त्रिहत्तर',
                                '74': 'चौहत्तर',
                                '75': 'पचहत्तर',
                                '76': 'छयहत्तर',
                                '77': 'सतहत्तर',
                                '78': 'अठहत्तर',
                                '79': 'उनासी',
                                '80': 'असी',
                                '81': 'एकासी',
                                '82': 'बयासी',
                                '83': 'त्रियासी',
                                '84': 'चौरासी',
                                '85': 'पचासी',
                                '86': 'छयासी',
                                '87': 'सतासी',
                                '88': 'अठासी',
                                '89': 'उनान्नब्बे',
                                '90': 'नब्बे',
                                '91': 'एकान्नब्बे',
                                '92': 'बयानब्बे',
                                '93': 'त्रियान्नब्बे',
                                '94': 'चौरान्नब्बे',
                                '95': 'पन्चानब्बे',
                                '96': 'छयान्नब्बे',
                                '97': 'सन्तान्नब्बे',
                                '98': 'अन्ठान्नब्बे',
                                '99': 'उनान्सय',
                                '100': 'एक सय',
                                '200': 'दुई सय',
                                '300': 'तीन सय',
                                '400': 'चार सय',
                                '500': 'पाँच सय',
                                '600': 'छ सय',
                                '700': 'सात सय',
                                '800': 'आठ सय',
                                '900': 'नौ सय',
                                '1,000': 'एक हजार',
                                '2,000': 'दुई हजार',
                                '3,000': 'तीन हजार',
                                '4,000': 'चार हजार',
                                '5,000': 'पाँच हजार',
                                '6,000': 'छ हजार',
                                '7,000': 'सात हजार',
                                '8,000': 'आठ हजार',
                                '9,000': 'नौ हजार',
                                '10,000': 'दश हजार',
                                '20,000': 'बिस हजार',
                                '30,000': 'तीस हजार',
                                '40,000': 'चालिस हजार',
                                '50,000': 'पचास हजार',
                                '60,000': 'साठी हजार',
                                '70,000': 'सत्तरी हजार',
                                '80,000': 'असी हजार',
                                '90,000': 'नब्बे हजार',
                                '100,000': 'एक लाख',
                                '1,000,000': 'दश लाख',
                                '10,000,000': 'एक करोड',
                                '1,000,000,000': 'एक अर्ब',
                                '100,000,000,000': 'एक खर्ब'}

    def merge(self, lpair, rpair):
        ltext, lnum = lpair
        rtext, rnum = rpair
        if lnum == 1 and rnum < 100:
            return (rtext, rnum)
        elif 100 > lnum > rnum:
            return ("%s %s" % (ltext, rtext), lnum + rnum)
        elif lnum >= 100 > rnum:
            return ("%s %s" % (ltext, rtext), lnum + rnum)
        elif rnum > lnum:
            return ("%s %s" % (ltext, rtext), lnum * rnum)
        return ("%s %s" % (ltext, rtext), lnum + rnum)

    def to_ordinal_num(self, value):
        self.verify_ordinal(value)
        return "%s%s" % (value, self.to_ordinal(value))

    def to_ordinal(self, value):
        self.verify_ordinal(value)
        try:
            ordinal_word = self.ords[value]
        except KeyError:
            ordinal_word = value #TODO: check nepali ordinal word suffix 
        return ordinal_word


    def to_year(self, val, suffix=None, longval=True):
        if val < 0:
            val = abs(val)
            suffix = 'BC' if not suffix else suffix
        high, low = (val // 100, val % 100)
        # If year is 00XX, X00X, or beyond 9999, go cardinal.
        if (high == 0
                or (high % 10 == 0 and low < 10)
                or high >= 100):
            valtext = self.to_cardinal(val)
        else:
            hightext = self.to_cardinal(high)
            if low == 0:
                lowtext = "सय"
            elif low < 10:
                lowtext = "%s" % self.to_cardinal(low)
            else:
                lowtext = self.to_cardinal(low)
            valtext = "%s %s" % (hightext, lowtext)
        return (valtext if not suffix
                else "%s %s" % (valtext, suffix))
