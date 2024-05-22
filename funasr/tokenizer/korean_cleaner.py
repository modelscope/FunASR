# Referenced from https://github.com/hccho2/Tacotron-Wavenet-Vocoder-Korean

import re


class KoreanCleaner:
    @classmethod
    def _normalize_numbers(cls, text):
        number_to_kor = {
            "0": "영",
            "1": "일",
            "2": "이",
            "3": "삼",
            "4": "사",
            "5": "오",
            "6": "육",
            "7": "칠",
            "8": "팔",
            "9": "구",
        }
        new_text = "".join(
            number_to_kor[char] if char in number_to_kor.keys() else char for char in text
        )
        return new_text

    @classmethod
    def _normalize_english_text(cls, text):
        upper_alphabet_to_kor = {
            "A": "에이",
            "B": "비",
            "C": "씨",
            "D": "디",
            "E": "이",
            "F": "에프",
            "G": "지",
            "H": "에이치",
            "I": "아이",
            "J": "제이",
            "K": "케이",
            "L": "엘",
            "M": "엠",
            "N": "엔",
            "O": "오",
            "P": "피",
            "Q": "큐",
            "R": "알",
            "S": "에스",
            "T": "티",
            "U": "유",
            "V": "브이",
            "W": "더블유",
            "X": "엑스",
            "Y": "와이",
            "Z": "지",
        }
        new_text = re.sub("[a-z]+", lambda x: str.upper(x.group()), text)
        new_text = "".join(
            upper_alphabet_to_kor[char] if char in upper_alphabet_to_kor.keys() else char
            for char in new_text
        )

        return new_text

    @classmethod
    def normalize_text(cls, text):
        # stage 0 : text strip
        text = text.strip()

        # stage 1 : normalize numbers
        text = cls._normalize_numbers(text)

        # stage 2 : normalize english text
        text = cls._normalize_english_text(text)
        return text
