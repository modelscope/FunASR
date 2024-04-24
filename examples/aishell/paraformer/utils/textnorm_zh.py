#!/usr/bin/env python3
# coding=utf-8

# Authors:
#   2019.5 Zhiyang Zhou (https://github.com/Joee1995/chn_text_norm.git)
#   2019.9 Jiayu DU
#
# requirements:
#   - python 3.X
# notes: python 2.X WILL fail or produce misleading results

import sys, os, argparse, codecs, string, re

# ================================================================================ #
#                                    basic constant
# ================================================================================ #
CHINESE_DIGIS = "零一二三四五六七八九"
BIG_CHINESE_DIGIS_SIMPLIFIED = "零壹贰叁肆伍陆柒捌玖"
BIG_CHINESE_DIGIS_TRADITIONAL = "零壹貳參肆伍陸柒捌玖"
SMALLER_BIG_CHINESE_UNITS_SIMPLIFIED = "十百千万"
SMALLER_BIG_CHINESE_UNITS_TRADITIONAL = "拾佰仟萬"
LARGER_CHINESE_NUMERING_UNITS_SIMPLIFIED = "亿兆京垓秭穰沟涧正载"
LARGER_CHINESE_NUMERING_UNITS_TRADITIONAL = "億兆京垓秭穰溝澗正載"
SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED = "十百千万"
SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL = "拾佰仟萬"

ZERO_ALT = "〇"
ONE_ALT = "幺"
TWO_ALTS = ["两", "兩"]

POSITIVE = ["正", "正"]
NEGATIVE = ["负", "負"]
POINT = ["点", "點"]
# PLUS = [u'加', u'加']
# SIL = [u'杠', u'槓']

FILLER_CHARS = ["呃", "啊"]
ER_WHITELIST = (
    "(儿女|儿子|儿孙|女儿|儿媳|妻儿|"
    "胎儿|婴儿|新生儿|婴幼儿|幼儿|少儿|小儿|儿歌|儿童|儿科|托儿所|孤儿|"
    "儿戏|儿化|台儿庄|鹿儿岛|正儿八经|吊儿郎当|生儿育女|托儿带女|养儿防老|痴儿呆女|"
    "佳儿佳妇|儿怜兽扰|儿无常父|儿不嫌母丑|儿行千里母担忧|儿大不由爷|苏乞儿)"
)

# 中文数字系统类型
NUMBERING_TYPES = ["low", "mid", "high"]

CURRENCY_NAMES = (
    "(人民币|美元|日元|英镑|欧元|马克|法郎|加拿大元|澳元|港币|先令|芬兰马克|爱尔兰镑|"
    "里拉|荷兰盾|埃斯库多|比塞塔|印尼盾|林吉特|新西兰元|比索|卢布|新加坡元|韩元|泰铢)"
)
CURRENCY_UNITS = (
    "((亿|千万|百万|万|千|百)|(亿|千万|百万|万|千|百|)元|(亿|千万|百万|万|千|百|)块|角|毛|分)"
)
COM_QUANTIFIERS = (
    "(匹|张|座|回|场|尾|条|个|首|阙|阵|网|炮|顶|丘|棵|只|支|袭|辆|挑|担|颗|壳|窠|曲|墙|群|腔|"
    "砣|座|客|贯|扎|捆|刀|令|打|手|罗|坡|山|岭|江|溪|钟|队|单|双|对|出|口|头|脚|板|跳|枝|件|贴|"
    "针|线|管|名|位|身|堂|课|本|页|家|户|层|丝|毫|厘|分|钱|两|斤|担|铢|石|钧|锱|忽|(千|毫|微)克|"
    "毫|厘|分|寸|尺|丈|里|寻|常|铺|程|(千|分|厘|毫|微)米|撮|勺|合|升|斗|石|盘|碗|碟|叠|桶|笼|盆|"
    "盒|杯|钟|斛|锅|簋|篮|盘|桶|罐|瓶|壶|卮|盏|箩|箱|煲|啖|袋|钵|年|月|日|季|刻|时|周|天|秒|分|旬|"
    "纪|岁|世|更|夜|春|夏|秋|冬|代|伏|辈|丸|泡|粒|颗|幢|堆|条|根|支|道|面|片|张|颗|块)"
)

# punctuation information are based on Zhon project (https://github.com/tsroten/zhon.git)
CHINESE_PUNC_STOP = "！？｡。"
CHINESE_PUNC_NON_STOP = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
CHINESE_PUNC_LIST = CHINESE_PUNC_STOP + CHINESE_PUNC_NON_STOP


# ================================================================================ #
#                                    basic class
# ================================================================================ #
class ChineseChar(object):
    """
    中文字符
    每个字符对应简体和繁体,
    e.g. 简体 = '负', 繁体 = '負'
    转换时可转换为简体或繁体
    """

    def __init__(self, simplified, traditional):
        self.simplified = simplified
        self.traditional = traditional
        # self.__repr__ = self.__str__

    def __str__(self):
        return self.simplified or self.traditional or None

    def __repr__(self):
        return self.__str__()


class ChineseNumberUnit(ChineseChar):
    """
    中文数字/数位字符
    每个字符除繁简体外还有一个额外的大写字符
    e.g. '陆' 和 '陸'
    """

    def __init__(self, power, simplified, traditional, big_s, big_t):
        super(ChineseNumberUnit, self).__init__(simplified, traditional)
        self.power = power
        self.big_s = big_s
        self.big_t = big_t

    def __str__(self):
        return "10^{}".format(self.power)

    @classmethod
    def create(cls, index, value, numbering_type=NUMBERING_TYPES[1], small_unit=False):

        if small_unit:
            return ChineseNumberUnit(
                power=index + 1,
                simplified=value[0],
                traditional=value[1],
                big_s=value[1],
                big_t=value[1],
            )
        elif numbering_type == NUMBERING_TYPES[0]:
            return ChineseNumberUnit(
                power=index + 8,
                simplified=value[0],
                traditional=value[1],
                big_s=value[0],
                big_t=value[1],
            )
        elif numbering_type == NUMBERING_TYPES[1]:
            return ChineseNumberUnit(
                power=(index + 2) * 4,
                simplified=value[0],
                traditional=value[1],
                big_s=value[0],
                big_t=value[1],
            )
        elif numbering_type == NUMBERING_TYPES[2]:
            return ChineseNumberUnit(
                power=pow(2, index + 3),
                simplified=value[0],
                traditional=value[1],
                big_s=value[0],
                big_t=value[1],
            )
        else:
            raise ValueError(
                "Counting type should be in {0} ({1} provided).".format(
                    NUMBERING_TYPES, numbering_type
                )
            )


class ChineseNumberDigit(ChineseChar):
    """
    中文数字字符
    """

    def __init__(self, value, simplified, traditional, big_s, big_t, alt_s=None, alt_t=None):
        super(ChineseNumberDigit, self).__init__(simplified, traditional)
        self.value = value
        self.big_s = big_s
        self.big_t = big_t
        self.alt_s = alt_s
        self.alt_t = alt_t

    def __str__(self):
        return str(self.value)

    @classmethod
    def create(cls, i, v):
        return ChineseNumberDigit(i, v[0], v[1], v[2], v[3])


class ChineseMath(ChineseChar):
    """
    中文数位字符
    """

    def __init__(self, simplified, traditional, symbol, expression=None):
        super(ChineseMath, self).__init__(simplified, traditional)
        self.symbol = symbol
        self.expression = expression
        self.big_s = simplified
        self.big_t = traditional


CC, CNU, CND, CM = ChineseChar, ChineseNumberUnit, ChineseNumberDigit, ChineseMath


class NumberSystem(object):
    """
    中文数字系统
    """

    pass


class MathSymbol(object):
    """
    用于中文数字系统的数学符号 (繁/简体), e.g.
    positive = ['正', '正']
    negative = ['负', '負']
    point = ['点', '點']
    """

    def __init__(self, positive, negative, point):
        self.positive = positive
        self.negative = negative
        self.point = point

    def __iter__(self):
        for v in self.__dict__.values():
            yield v


# class OtherSymbol(object):
#     """
#     其他符号
#     """
#
#     def __init__(self, sil):
#         self.sil = sil
#
#     def __iter__(self):
#         for v in self.__dict__.values():
#             yield v


# ================================================================================ #
#                                    basic utils
# ================================================================================ #
def create_system(numbering_type=NUMBERING_TYPES[1]):
    """
    根据数字系统类型返回创建相应的数字系统，默认为 mid
    NUMBERING_TYPES = ['low', 'mid', 'high']: 中文数字系统类型
        low:  '兆' = '亿' * '十' = $10^{9}$,  '京' = '兆' * '十', etc.
        mid:  '兆' = '亿' * '万' = $10^{12}$, '京' = '兆' * '万', etc.
        high: '兆' = '亿' * '亿' = $10^{16}$, '京' = '兆' * '兆', etc.
    返回对应的数字系统
    """

    # chinese number units of '亿' and larger
    all_larger_units = zip(
        LARGER_CHINESE_NUMERING_UNITS_SIMPLIFIED, LARGER_CHINESE_NUMERING_UNITS_TRADITIONAL
    )
    larger_units = [CNU.create(i, v, numbering_type, False) for i, v in enumerate(all_larger_units)]
    # chinese number units of '十, 百, 千, 万'
    all_smaller_units = zip(
        SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED, SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL
    )
    smaller_units = [CNU.create(i, v, small_unit=True) for i, v in enumerate(all_smaller_units)]
    # digis
    chinese_digis = zip(
        CHINESE_DIGIS, CHINESE_DIGIS, BIG_CHINESE_DIGIS_SIMPLIFIED, BIG_CHINESE_DIGIS_TRADITIONAL
    )
    digits = [CND.create(i, v) for i, v in enumerate(chinese_digis)]
    digits[0].alt_s, digits[0].alt_t = ZERO_ALT, ZERO_ALT
    digits[1].alt_s, digits[1].alt_t = ONE_ALT, ONE_ALT
    digits[2].alt_s, digits[2].alt_t = TWO_ALTS[0], TWO_ALTS[1]

    # symbols
    positive_cn = CM(POSITIVE[0], POSITIVE[1], "+", lambda x: x)
    negative_cn = CM(NEGATIVE[0], NEGATIVE[1], "-", lambda x: -x)
    point_cn = CM(POINT[0], POINT[1], ".", lambda x, y: float(str(x) + "." + str(y)))
    # sil_cn = CM(SIL[0], SIL[1], '-', lambda x, y: float(str(x) + '-' + str(y)))
    system = NumberSystem()
    system.units = smaller_units + larger_units
    system.digits = digits
    system.math = MathSymbol(positive_cn, negative_cn, point_cn)
    # system.symbols = OtherSymbol(sil_cn)
    return system


def chn2num(chinese_string, numbering_type=NUMBERING_TYPES[1]):

    def get_symbol(char, system):
        for u in system.units:
            if char in [u.traditional, u.simplified, u.big_s, u.big_t]:
                return u
        for d in system.digits:
            if char in [d.traditional, d.simplified, d.big_s, d.big_t, d.alt_s, d.alt_t]:
                return d
        for m in system.math:
            if char in [m.traditional, m.simplified]:
                return m

    def string2symbols(chinese_string, system):
        int_string, dec_string = chinese_string, ""
        for p in [system.math.point.simplified, system.math.point.traditional]:
            if p in chinese_string:
                int_string, dec_string = chinese_string.split(p)
                break
        return [get_symbol(c, system) for c in int_string], [
            get_symbol(c, system) for c in dec_string
        ]

    def correct_symbols(integer_symbols, system):
        """
        一百八 to 一百八十
        一亿一千三百万 to 一亿 一千万 三百万
        """

        if integer_symbols and isinstance(integer_symbols[0], CNU):
            if integer_symbols[0].power == 1:
                integer_symbols = [system.digits[1]] + integer_symbols

        if len(integer_symbols) > 1:
            if isinstance(integer_symbols[-1], CND) and isinstance(integer_symbols[-2], CNU):
                integer_symbols.append(CNU(integer_symbols[-2].power - 1, None, None, None, None))

        result = []
        unit_count = 0
        for s in integer_symbols:
            if isinstance(s, CND):
                result.append(s)
                unit_count = 0
            elif isinstance(s, CNU):
                current_unit = CNU(s.power, None, None, None, None)
                unit_count += 1

            if unit_count == 1:
                result.append(current_unit)
            elif unit_count > 1:
                for i in range(len(result)):
                    if (
                        isinstance(result[-i - 1], CNU)
                        and result[-i - 1].power < current_unit.power
                    ):
                        result[-i - 1] = CNU(
                            result[-i - 1].power + current_unit.power, None, None, None, None
                        )
        return result

    def compute_value(integer_symbols):
        """
        Compute the value.
        When current unit is larger than previous unit, current unit * all previous units will be used as all previous units.
        e.g. '两千万' = 2000 * 10000 not 2000 + 10000
        """
        value = [0]
        last_power = 0
        for s in integer_symbols:
            if isinstance(s, CND):
                value[-1] = s.value
            elif isinstance(s, CNU):
                value[-1] *= pow(10, s.power)
                if s.power > last_power:
                    value[:-1] = list(map(lambda v: v * pow(10, s.power), value[:-1]))
                    last_power = s.power
                value.append(0)
        return sum(value)

    system = create_system(numbering_type)
    int_part, dec_part = string2symbols(chinese_string, system)
    int_part = correct_symbols(int_part, system)
    int_str = str(compute_value(int_part))
    dec_str = "".join([str(d.value) for d in dec_part])
    if dec_part:
        return "{0}.{1}".format(int_str, dec_str)
    else:
        return int_str


def num2chn(
    number_string,
    numbering_type=NUMBERING_TYPES[1],
    big=False,
    traditional=False,
    alt_zero=False,
    alt_one=False,
    alt_two=True,
    use_zeros=True,
    use_units=True,
):

    def get_value(value_string, use_zeros=True):

        striped_string = value_string.lstrip("0")

        # record nothing if all zeros
        if not striped_string:
            return []

        # record one digits
        elif len(striped_string) == 1:
            if use_zeros and len(value_string) != len(striped_string):
                return [system.digits[0], system.digits[int(striped_string)]]
            else:
                return [system.digits[int(striped_string)]]

        # recursively record multiple digits
        else:
            result_unit = next(u for u in reversed(system.units) if u.power < len(striped_string))
            result_string = value_string[: -result_unit.power]
            return (
                get_value(result_string)
                + [result_unit]
                + get_value(striped_string[-result_unit.power :])
            )

    system = create_system(numbering_type)

    int_dec = number_string.split(".")
    if len(int_dec) == 1:
        int_string = int_dec[0]
        dec_string = ""
    elif len(int_dec) == 2:
        int_string = int_dec[0]
        dec_string = int_dec[1]
    else:
        raise ValueError(
            "invalid input num string with more than one dot: {}".format(number_string)
        )

    if use_units and len(int_string) > 1:
        result_symbols = get_value(int_string)
    else:
        result_symbols = [system.digits[int(c)] for c in int_string]
    dec_symbols = [system.digits[int(c)] for c in dec_string]
    if dec_string:
        result_symbols += [system.math.point] + dec_symbols

    if alt_two:
        liang = CND(
            2,
            system.digits[2].alt_s,
            system.digits[2].alt_t,
            system.digits[2].big_s,
            system.digits[2].big_t,
        )
        for i, v in enumerate(result_symbols):
            if isinstance(v, CND) and v.value == 2:
                next_symbol = result_symbols[i + 1] if i < len(result_symbols) - 1 else None
                previous_symbol = result_symbols[i - 1] if i > 0 else None
                if isinstance(next_symbol, CNU) and isinstance(previous_symbol, (CNU, type(None))):
                    if next_symbol.power != 1 and (
                        (previous_symbol is None) or (previous_symbol.power != 1)
                    ):
                        result_symbols[i] = liang

    # if big is True, '两' will not be used and `alt_two` has no impact on output
    if big:
        attr_name = "big_"
        if traditional:
            attr_name += "t"
        else:
            attr_name += "s"
    else:
        if traditional:
            attr_name = "traditional"
        else:
            attr_name = "simplified"

    result = "".join([getattr(s, attr_name) for s in result_symbols])

    # if not use_zeros:
    #     result = result.strip(getattr(system.digits[0], attr_name))

    if alt_zero:
        result = result.replace(getattr(system.digits[0], attr_name), system.digits[0].alt_s)

    if alt_one:
        result = result.replace(getattr(system.digits[1], attr_name), system.digits[1].alt_s)

    for i, p in enumerate(POINT):
        if result.startswith(p):
            return CHINESE_DIGIS[0] + result

    # ^10, 11, .., 19
    if (
        len(result) >= 2
        and result[1]
        in [
            SMALLER_CHINESE_NUMERING_UNITS_SIMPLIFIED[0],
            SMALLER_CHINESE_NUMERING_UNITS_TRADITIONAL[0],
        ]
        and result[0]
        in [CHINESE_DIGIS[1], BIG_CHINESE_DIGIS_SIMPLIFIED[1], BIG_CHINESE_DIGIS_TRADITIONAL[1]]
    ):
        result = result[1:]

    return result


# ================================================================================ #
#                          different types of rewriters
# ================================================================================ #
class Cardinal:
    """
    CARDINAL类
    """

    def __init__(self, cardinal=None, chntext=None):
        self.cardinal = cardinal
        self.chntext = chntext

    def chntext2cardinal(self):
        return chn2num(self.chntext)

    def cardinal2chntext(self):
        return num2chn(self.cardinal)


class Digit:
    """
    DIGIT类
    """

    def __init__(self, digit=None, chntext=None):
        self.digit = digit
        self.chntext = chntext

    # def chntext2digit(self):
    #     return chn2num(self.chntext)

    def digit2chntext(self):
        return num2chn(self.digit, alt_two=False, use_units=False)


class TelePhone:
    """
    TELEPHONE类
    """

    def __init__(self, telephone=None, raw_chntext=None, chntext=None):
        self.telephone = telephone
        self.raw_chntext = raw_chntext
        self.chntext = chntext

    # def chntext2telephone(self):
    #     sil_parts = self.raw_chntext.split('<SIL>')
    #     self.telephone = '-'.join([
    #         str(chn2num(p)) for p in sil_parts
    #     ])
    #     return self.telephone

    def telephone2chntext(self, fixed=False):

        if fixed:
            sil_parts = self.telephone.split("-")
            self.raw_chntext = "<SIL>".join(
                [num2chn(part, alt_two=False, use_units=False) for part in sil_parts]
            )
            self.chntext = self.raw_chntext.replace("<SIL>", "")
        else:
            sp_parts = self.telephone.strip("+").split()
            self.raw_chntext = "<SP>".join(
                [num2chn(part, alt_two=False, use_units=False) for part in sp_parts]
            )
            self.chntext = self.raw_chntext.replace("<SP>", "")
        return self.chntext


class Fraction:
    """
    FRACTION类
    """

    def __init__(self, fraction=None, chntext=None):
        self.fraction = fraction
        self.chntext = chntext

    def chntext2fraction(self):
        denominator, numerator = self.chntext.split("分之")
        return chn2num(numerator) + "/" + chn2num(denominator)

    def fraction2chntext(self):
        numerator, denominator = self.fraction.split("/")
        return num2chn(denominator) + "分之" + num2chn(numerator)


class Date:
    """
    DATE类
    """

    def __init__(self, date=None, chntext=None):
        self.date = date
        self.chntext = chntext

    # def chntext2date(self):
    #     chntext = self.chntext
    #     try:
    #         year, other = chntext.strip().split('年', maxsplit=1)
    #         year = Digit(chntext=year).digit2chntext() + '年'
    #     except ValueError:
    #         other = chntext
    #         year = ''
    #     if other:
    #         try:
    #             month, day = other.strip().split('月', maxsplit=1)
    #             month = Cardinal(chntext=month).chntext2cardinal() + '月'
    #         except ValueError:
    #             day = chntext
    #             month = ''
    #         if day:
    #             day = Cardinal(chntext=day[:-1]).chntext2cardinal() + day[-1]
    #     else:
    #         month = ''
    #         day = ''
    #     date = year + month + day
    #     self.date = date
    #     return self.date

    def date2chntext(self):
        date = self.date
        try:
            year, other = date.strip().split("年", 1)
            year = Digit(digit=year).digit2chntext() + "年"
        except ValueError:
            other = date
            year = ""
        if other:
            try:
                month, day = other.strip().split("月", 1)
                month = Cardinal(cardinal=month).cardinal2chntext() + "月"
            except ValueError:
                day = date
                month = ""
            if day:
                day = Cardinal(cardinal=day[:-1]).cardinal2chntext() + day[-1]
        else:
            month = ""
            day = ""
        chntext = year + month + day
        self.chntext = chntext
        return self.chntext


class Money:
    """
    MONEY类
    """

    def __init__(self, money=None, chntext=None):
        self.money = money
        self.chntext = chntext

    # def chntext2money(self):
    #     return self.money

    def money2chntext(self):
        money = self.money
        pattern = re.compile(r"(\d+(\.\d+)?)")
        matchers = pattern.findall(money)
        if matchers:
            for matcher in matchers:
                money = money.replace(matcher[0], Cardinal(cardinal=matcher[0]).cardinal2chntext())
        self.chntext = money
        return self.chntext


class Percentage:
    """
    PERCENTAGE类
    """

    def __init__(self, percentage=None, chntext=None):
        self.percentage = percentage
        self.chntext = chntext

    def chntext2percentage(self):
        return chn2num(self.chntext.strip().strip("百分之")) + "%"

    def percentage2chntext(self):
        return "百分之" + num2chn(self.percentage.strip().strip("%"))


def remove_erhua(text, er_whitelist):
    """
    去除儿化音词中的儿:
    他女儿在那边儿 -> 他女儿在那边
    """

    er_pattern = re.compile(er_whitelist)
    new_str = ""
    while re.search("儿", text):
        a = re.search("儿", text).span()
        remove_er_flag = 0

        if er_pattern.search(text):
            b = er_pattern.search(text).span()
            if b[0] <= a[0]:
                remove_er_flag = 1

        if remove_er_flag == 0:
            new_str = new_str + text[0 : a[0]]
            text = text[a[1] :]
        else:
            new_str = new_str + text[0 : b[1]]
            text = text[b[1] :]

    text = new_str + text
    return text


# ================================================================================ #
#                            NSW Normalizer
# ================================================================================ #
class NSWNormalizer:
    def __init__(self, raw_text):
        self.raw_text = "^" + raw_text + "$"
        self.norm_text = ""

    def _particular(self):
        text = self.norm_text
        pattern = re.compile(r"(([a-zA-Z]+)二([a-zA-Z]+))")
        matchers = pattern.findall(text)
        if matchers:
            # print('particular')
            for matcher in matchers:
                text = text.replace(matcher[0], matcher[1] + "2" + matcher[2], 1)
        self.norm_text = text
        return self.norm_text

    def normalize(self):
        text = self.raw_text

        # 规范化日期
        pattern = re.compile(r"\D+((([089]\d|(19|20)\d{2})年)?(\d{1,2}月(\d{1,2}[日号])?)?)")
        matchers = pattern.findall(text)
        if matchers:
            # print('date')
            for matcher in matchers:
                text = text.replace(matcher[0], Date(date=matcher[0]).date2chntext(), 1)

        # 规范化金钱
        pattern = re.compile(
            r"\D+((\d+(\.\d+)?)[多余几]?" + CURRENCY_UNITS + r"(\d" + CURRENCY_UNITS + r"?)?)"
        )
        matchers = pattern.findall(text)
        if matchers:
            # print('money')
            for matcher in matchers:
                text = text.replace(matcher[0], Money(money=matcher[0]).money2chntext(), 1)

        # 规范化固话/手机号码
        # 手机
        # http://www.jihaoba.com/news/show/13680
        # 移动：139、138、137、136、135、134、159、158、157、150、151、152、188、187、182、183、184、178、198
        # 联通：130、131、132、156、155、186、185、176
        # 电信：133、153、189、180、181、177
        pattern = re.compile(r"\D((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})\D")
        matchers = pattern.findall(text)
        if matchers:
            # print('telephone')
            for matcher in matchers:
                text = text.replace(
                    matcher[0], TelePhone(telephone=matcher[0]).telephone2chntext(), 1
                )
        # 固话
        pattern = re.compile(r"\D((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})\D")
        matchers = pattern.findall(text)
        if matchers:
            # print('fixed telephone')
            for matcher in matchers:
                text = text.replace(
                    matcher[0], TelePhone(telephone=matcher[0]).telephone2chntext(fixed=True), 1
                )

        # 规范化分数
        pattern = re.compile(r"(\d+/\d+)")
        matchers = pattern.findall(text)
        if matchers:
            # print('fraction')
            for matcher in matchers:
                text = text.replace(matcher, Fraction(fraction=matcher).fraction2chntext(), 1)

        # 规范化百分数
        text = text.replace("％", "%")
        pattern = re.compile(r"(\d+(\.\d+)?%)")
        matchers = pattern.findall(text)
        if matchers:
            # print('percentage')
            for matcher in matchers:
                text = text.replace(
                    matcher[0], Percentage(percentage=matcher[0]).percentage2chntext(), 1
                )

        # 规范化纯数+量词
        pattern = re.compile(r"(\d+(\.\d+)?)[多余几]?" + COM_QUANTIFIERS)
        matchers = pattern.findall(text)
        if matchers:
            # print('cardinal+quantifier')
            for matcher in matchers:
                text = text.replace(matcher[0], Cardinal(cardinal=matcher[0]).cardinal2chntext(), 1)

        # 规范化数字编号
        pattern = re.compile(r"(\d{4,32})")
        matchers = pattern.findall(text)
        if matchers:
            # print('digit')
            for matcher in matchers:
                text = text.replace(matcher, Digit(digit=matcher).digit2chntext(), 1)

        # 规范化纯数
        pattern = re.compile(r"(\d+(\.\d+)?)")
        matchers = pattern.findall(text)
        if matchers:
            # print('cardinal')
            for matcher in matchers:
                text = text.replace(matcher[0], Cardinal(cardinal=matcher[0]).cardinal2chntext(), 1)

        self.norm_text = text
        self._particular()

        return self.norm_text.lstrip("^").rstrip("$")


def nsw_test_case(raw_text):
    print("I:" + raw_text)
    print("O:" + NSWNormalizer(raw_text).normalize())
    print("")


def nsw_test():
    nsw_test_case("固话：0595-23865596或23880880。")
    nsw_test_case("固话：0595-23865596或23880880。")
    nsw_test_case("手机：+86 19859213959或15659451527。")
    nsw_test_case("分数：32477/76391。")
    nsw_test_case("百分数：80.03%。")
    nsw_test_case("编号：31520181154418。")
    nsw_test_case("纯数：2983.07克或12345.60米。")
    nsw_test_case("日期：1999年2月20日或09年3月15号。")
    nsw_test_case("金钱：12块5，34.5元，20.1万")
    nsw_test_case("特殊：O2O或B2C。")
    nsw_test_case("3456万吨")
    nsw_test_case("2938个")
    nsw_test_case("938")
    nsw_test_case("今天吃了115个小笼包231个馒头")
    nsw_test_case("有62％的概率")


if __name__ == "__main__":
    # nsw_test()

    p = argparse.ArgumentParser()
    p.add_argument("ifile", help="input filename, assume utf-8 encoding")
    p.add_argument("ofile", help="output filename")
    p.add_argument("--to_upper", action="store_true", help="convert to upper case")
    p.add_argument("--to_lower", action="store_true", help="convert to lower case")
    p.add_argument(
        "--has_key", action="store_true", help="input text has Kaldi's key as first field."
    )
    p.add_argument(
        "--remove_fillers", type=bool, default=True, help='remove filler chars such as "呃, 啊"'
    )
    p.add_argument(
        "--remove_erhua", type=bool, default=True, help='remove erhua chars such as "这儿"'
    )
    p.add_argument(
        "--log_interval", type=int, default=10000, help="log interval in number of processed lines"
    )
    args = p.parse_args()

    ifile = codecs.open(args.ifile, "r", "utf8")
    ofile = codecs.open(args.ofile, "w+", "utf8")

    n = 0
    for l in ifile:
        key = ""
        text = ""
        if args.has_key:
            cols = l.split(maxsplit=1)
            key = cols[0]
            if len(cols) == 2:
                text = cols[1].strip()
            else:
                text = ""
        else:
            text = l.strip()

        # cases
        if args.to_upper and args.to_lower:
            sys.stderr.write("text norm: to_upper OR to_lower?")
            exit(1)
        if args.to_upper:
            text = text.upper()
        if args.to_lower:
            text = text.lower()

        # Filler chars removal
        if args.remove_fillers:
            for ch in FILLER_CHARS:
                text = text.replace(ch, "")

        if args.remove_erhua:
            text = remove_erhua(text, ER_WHITELIST)

        # NSW(Non-Standard-Word) normalization
        text = NSWNormalizer(text).normalize()

        # Punctuations removal
        old_chars = CHINESE_PUNC_LIST + string.punctuation  # includes all CN and EN punctuations
        new_chars = " " * len(old_chars)
        del_chars = ""
        text = text.translate(str.maketrans(old_chars, new_chars, del_chars))

        #
        if args.has_key:
            ofile.write(key + "\t" + text + "\n")
        else:
            ofile.write(text + "\n")

        n += 1
        if n % args.log_interval == 0:
            sys.stderr.write("text norm: {} lines done.\n".format(n))

    sys.stderr.write("text norm: {} lines done in total.\n".format(n))

    ifile.close()
    ofile.close()
