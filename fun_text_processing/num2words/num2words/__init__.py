# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from . import (lang_AR, lang_AZ, lang_BN, lang_BG, lang_CA, lang_CZ, lang_DE, lang_DK, 
                lang_EN, lang_EL, lang_EN_IN,
               lang_ES, lang_ES_CO, lang_ES_NI, lang_ES_VE, lang_FI, lang_FR,
               lang_FR_BE, lang_FR_CH, lang_FR_DZ, lang_HE, lang_HU, lang_ID,
               lang_IT, lang_IW, lang_JA, lang_KH, lang_KN, lang_KO, lang_KZ, lang_LT,
               lang_LO, lang_LV, lang_MN, lang_MY,
               lang_NE, lang_NL, lang_NO, lang_PL, lang_PT, lang_PT_BR, lang_RO,
               lang_RU, lang_SI, lang_SL, lang_SR, lang_SV, lang_SW, lang_TE, lang_TH, lang_TR,
               lang_UK, lang_UR, lang_VI)

CONVERTER_CLASSES = {
    'ar': lang_AR.Num2Word_AR(),
    'az': lang_AZ.Num2Word_AZ(),
    'bn': lang_BN.Num2Word_BN(),
    'bg': lang_BG.Num2Word_BG(),
    'ca': lang_CA.Num2Word_CA(),
    'cz': lang_CZ.Num2Word_CZ(),
    'en': lang_EN.Num2Word_EN(),
    'el': lang_EL.Num2Word_EL(),
    'en_IN': lang_EN_IN.Num2Word_EN_IN(),
    'fr': lang_FR.Num2Word_FR(),
    'fr_CH': lang_FR_CH.Num2Word_FR_CH(),
    'fr_BE': lang_FR_BE.Num2Word_FR_BE(),
    'fr_DZ': lang_FR_DZ.Num2Word_FR_DZ(),
    'de': lang_DE.Num2Word_DE(),
    'fi': lang_FI.Num2Word_FI(),
    'es': lang_ES.Num2Word_ES(),
    'es_CO': lang_ES_CO.Num2Word_ES_CO(),
    'es_NI': lang_ES_NI.Num2Word_ES_NI(),
    'es_VE': lang_ES_VE.Num2Word_ES_VE(),
    'id': lang_ID.Num2Word_ID(),
    'iw': lang_IW.Num2Word_IW(),
    'ja': lang_JA.Num2Word_JA(),
    'kh': lang_KH.Num2Word_KH(),
    'kn': lang_KN.Num2Word_KN(),
    'ko': lang_KO.Num2Word_KO(),
    'kz': lang_KZ.Num2Word_KZ(),
    'lo': lang_LO.Num2Word_LO(),
    'lt': lang_LT.Num2Word_LT(),
    'lv': lang_LV.Num2Word_LV(),
    'mn': lang_MN.Num2Word_MN(),
    'my': lang_MY.Num2Word_MY(),
    'pl': lang_PL.Num2Word_PL(),
    'ro': lang_RO.Num2Word_RO(),
    'ru': lang_RU.Num2Word_RU(),
    'si': lang_SI.Num2Word_SI(),
    'sl': lang_SL.Num2Word_SL(),
    'sr': lang_SR.Num2Word_SR(),
    'sv': lang_SV.Num2Word_SV(),
    'sw': lang_SW.Num2Word_SW(),
    'no': lang_NO.Num2Word_NO(),
    'dk': lang_DK.Num2Word_DK(),
    'pt': lang_PT.Num2Word_PT(),
    'pt_BR': lang_PT_BR.Num2Word_PT_BR(),
    'he': lang_HE.Num2Word_HE(),
    'it': lang_IT.Num2Word_IT(),
    'vi': lang_VI.Num2Word_VI(),
    'th': lang_TH.Num2Word_TH(),
    'tr': lang_TR.Num2Word_TR(),
    'ne': lang_NE.Num2Word_NE(),
    'nl': lang_NL.Num2Word_NL(),
    'uk': lang_UK.Num2Word_UK(),
    'ur': lang_UR.Num2Word_UR(),
    'te': lang_TE.Num2Word_TE(),
    'hu': lang_HU.Num2Word_HU()
}

CONVERTES_TYPES = ['cardinal', 'ordinal', 'ordinal_num', 'year', 'currency']

'ar': lang_AR.Num2Word_AR(),
'az': lang_AZ.Num2Word_AZ(),
'bn': lang_BN.Num2Word_BN(),
'bg': lang_BG.Num2Word_BG(),
'ca': lang_CA.Num2Word_CA(),
'cz': lang_CZ.Num2Word_CZ(),
'en': lang_EN.Num2Word_EN(),
'el': lang_EL.Num2Word_EL(),
'en_IN': lang_EN_IN.Num2Word_EN_IN(),
'fr': lang_FR.Num2Word_FR(),
'fr_CH': lang_FR_CH.Num2Word_FR_CH(),
'fr_BE': lang_FR_BE.Num2Word_FR_BE(),
'fr_DZ': lang_FR_DZ.Num2Word_FR_DZ(),
'de': lang_DE.Num2Word_DE(),
'fi': lang_FI.Num2Word_FI(),
'es': lang_ES.Num2Word_ES(),
'es_CO': lang_ES_CO.Num2Word_ES_CO(),
'es_NI': lang_ES_NI.Num2Word_ES_NI(),
'es_VE': lang_ES_VE.Num2Word_ES_VE(),
'id': lang_ID.Num2Word_ID(),
'iw': lang_IW.Num2Word_IW(),
'ja': lang_JA.Num2Word_JA(),
'kh': lang_KH.Num2Word_KH(),
'kn': lang_KN.Num2Word_KN(),
'ko': lang_KO.Num2Word_KO(),
'kz': lang_KZ.Num2Word_KZ(),
'lo': lang_LO.Num2Word_LO(),
'lt': lang_LT.Num2Word_LT(),
'lv': lang_LV.Num2Word_LV(),
'mn': lang_MN.Num2Word_MN(),
'my': lang_MY.Num2Word_MY(),
'pl': lang_PL.Num2Word_PL(),
'ro': lang_RO.Num2Word_RO(),
'ru': lang_RU.Num2Word_RU(),
'si': lang_SI.Num2Word_SI(),
'sl': lang_SL.Num2Word_SL(),
'sr': lang_SR.Num2Word_SR(),
'sv': lang_SV.Num2Word_SV(),
'sw': lang_SW.Num2Word_SW(),
'no': lang_NO.Num2Word_NO(),
'dk': lang_DK.Num2Word_DK(),
'pt': lang_PT.Num2Word_PT(),
'pt_BR': lang_PT_BR.Num2Word_PT_BR(),
'he': lang_HE.Num2Word_HE(),
'it': lang_IT.Num2Word_IT(),
'vi': lang_VI.Num2Word_VI(),
'th': lang_TH.Num2Word_TH(),
'tr': lang_TR.Num2Word_TR(),
'ne': lang_NE.Num2Word_NE(),
'nl': lang_NL.Num2Word_NL(),
'uk': lang_UK.Num2Word_UK(),
'ur': lang_UR.Num2Word_UR(),
'te': lang_TE.Num2Word_TE(),
'hu': lang_HU.Num2Word_HU()
def num2words(number, ordinal=False, lang='en', to='cardinal', **kwargs):
    # We try the full language first
    if lang not in CONVERTER_CLASSES:
        # ... and then try only the first 2 letters
        lang = lang[:2]
    if lang not in CONVERTER_CLASSES:
        raise NotImplementedError()
    converter = CONVERTER_CLASSES[lang]

    if isinstance(number, str):
        number = converter.str_to_number(number)

    # backwards compatible
    if ordinal:
        return converter.to_ordinal(number)

    if to not in CONVERTES_TYPES:
        raise NotImplementedError()

    return getattr(converter, 'to_{}'.format(to))(number, **kwargs)
