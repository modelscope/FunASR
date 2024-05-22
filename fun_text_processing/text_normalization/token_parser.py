import string
from collections import OrderedDict
from typing import Dict, List, Union

PRESERVE_ORDER_KEY = "preserve_order"
EOS = "<EOS>"


class TokenParser:
    """
    Parses tokenized/classified text, e.g. 'tokens { money { integer: "20" currency: "$" } } tokens { name: "left"}'

    Args
        text: tokenized text
    """

    def __call__(self, text):
        """
        Setup function

        Args:
            text: text to be parsed

        """
        self.text = text
        self.len_text = len(text)
        self.char = text[0]  # cannot handle empty string
        self.index = 0

    def parse(self) -> List[dict]:
        """
        Main function. Implements grammar:
        A -> space F space F space F ... space

        Returns list of dictionaries
        """
        l = list()
        while self.parse_ws():
            token = self.parse_token()
            if not token:
                break
            l.append(token)
        return l

    def parse_token(self) -> Dict[str, Union[str, dict]]:
        """
        Implements grammar:
        F-> no_space KG no_space

        Returns: K, G as dictionary values
        """
        d = OrderedDict()
        key = self.parse_string_key()
        if key is None:
            return None
        self.parse_ws()
        if key == PRESERVE_ORDER_KEY:
            self.parse_char(":")
            self.parse_ws()
            value = self.parse_chars("true")
        else:
            value = self.parse_token_value()

        d[key] = value
        return d

    def parse_token_value(self) -> Union[str, dict]:
        """
        Implements grammar:
        G-> no_space :"VALUE" no_space | no_space {A} no_space

        Returns: string or dictionary
        """
        if self.char == ":":
            self.parse_char(":")
            self.parse_ws()
            self.parse_char('"')
            value_string = self.parse_string_value()
            self.parse_char('"')
            return value_string
        elif self.char == "{":
            d = OrderedDict()
            self.parse_char("{")
            list_token_dicts = self.parse()
            # flatten tokens
            for tok_dict in list_token_dicts:
                for k, v in tok_dict.items():
                    d[k] = v
            self.parse_char("}")
            return d
        else:
            raise ValueError()

    def parse_char(self, exp) -> bool:
        """
        Parses character

        Args:
            exp: character to read in

        Returns true if successful
        """
        assert self.char == exp
        self.read()
        return True

    def parse_chars(self, exp) -> bool:
        """
        Parses characters

        Args:
            exp: characters to read in

        Returns true if successful
        """
        ok = False
        for x in exp:
            ok |= self.parse_char(x)
        return ok

    def parse_string_key(self) -> str:
        """
        Parses string key, can only contain ascii and '_' characters

        Returns parsed string key
        """
        assert self.char not in string.whitespace and self.char != EOS
        incl_criterium = string.ascii_letters + "_"
        l = []
        while self.char in incl_criterium:
            l.append(self.char)
            if not self.read():
                raise ValueError()

        if not l:
            return None
        return "".join(l)

    def parse_string_value(self) -> str:
        """
        Parses string value, ends with quote followed by space

        Returns parsed string value
        """
        # assert self.char not in string.whitespace and self.char != EOS
        assert self.char != EOS
        l = []
        while self.char != '"' or self.text[self.index + 1] != " ":
            l.append(self.char)
            if not self.read():
                raise ValueError()

        if not l:
            return None
        return "".join(l)

    def parse_ws(self):
        """
        Deletes whitespaces.

        Returns true if not EOS after parsing
        """
        not_eos = self.char != EOS
        while not_eos and self.char == " ":
            not_eos = self.read()
        return not_eos

    def read(self):
        """
        Reads in next char.

        Returns true if not EOS
        """
        if self.index < self.len_text - 1:  # should be unique
            self.index += 1
            self.char = self.text[self.index]
            return True
        self.char = EOS
        return False
