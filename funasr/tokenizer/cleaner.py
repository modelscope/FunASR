from typing import Collection

from jaconv import jaconv

# import tacotron_cleaner.cleaners

try:
    from vietnamese_cleaner import vietnamese_cleaners
except ImportError:
    vietnamese_cleaners = None


class TextCleaner:
    """Text cleaner.

    Examples:
        >>> cleaner = TextCleaner("tacotron")
        >>> cleaner("(Hello-World);   &  jr. & dr.")
        'HELLO WORLD, AND JUNIOR AND DOCTOR'

    """

    def __init__(self, cleaner_types: Collection[str] = None):

        if cleaner_types is None:
            self.cleaner_types = []
        elif isinstance(cleaner_types, str):
            self.cleaner_types = [cleaner_types]
        else:
            self.cleaner_types = list(cleaner_types)

    def __call__(self, text: str) -> str:
        for t in self.cleaner_types:
            if t == "tacotron":
                # text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
                pass
            elif t == "jaconv":
                text = jaconv.normalize(text)
            elif t == "vietnamese":
                if vietnamese_cleaners is None:
                    raise RuntimeError("Please install underthesea")
                text = vietnamese_cleaners.vietnamese_cleaner(text)
            elif t == "korean_cleaner":
                text = KoreanCleaner.normalize_text(text)
            else:
                raise RuntimeError(f"Not supported: type={t}")

        return text
