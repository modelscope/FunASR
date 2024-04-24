import json
import re
import string
from collections import defaultdict, namedtuple
from typing import Dict, List, Optional, Set, Tuple
from unicodedata import category

import logging


EOS_TYPE = "EOS"
PUNCT_TYPE = "PUNCT"
PLAIN_TYPE = "PLAIN"
Instance = namedtuple("Instance", "token_type un_normalized normalized")
known_types = [
    "PLAIN",
    "DATE",
    "CARDINAL",
    "LETTERS",
    "VERBATIM",
    "MEASURE",
    "DECIMAL",
    "ORDINAL",
    "DIGIT",
    "MONEY",
    "TELEPHONE",
    "ELECTRONIC",
    "FRACTION",
    "TIME",
    "ADDRESS",
]


def _load_kaggle_text_norm_file(file_path: str) -> List[Instance]:
    """
    https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish
    Loads text file in the Kaggle Google text normalization file format: <semiotic class>\t<unnormalized text>\t<`self` if trivial class or normalized text>
    E.g.
    PLAIN   Brillantaisia   <self>
    PLAIN   is      <self>
    PLAIN   a       <self>
    PLAIN   genus   <self>
    PLAIN   of      <self>
    PLAIN   plant   <self>
    PLAIN   in      <self>
    PLAIN   family  <self>
    PLAIN   Acanthaceae     <self>
    PUNCT   .       sil
    <eos>   <eos>

    Args:
        file_path: file path to text file

    Returns: flat list of instances
    """
    res = []
    with open(file_path, "r") as fp:
        for line in fp:
            parts = line.strip().split("\t")
            if parts[0] == "<eos>":
                res.append(Instance(token_type=EOS_TYPE, un_normalized="", normalized=""))
            else:
                l_type, l_token, l_normalized = parts
                l_token = l_token.lower()
                l_normalized = l_normalized.lower()

                if l_type == PLAIN_TYPE:
                    res.append(
                        Instance(token_type=l_type, un_normalized=l_token, normalized=l_token)
                    )
                elif l_type != PUNCT_TYPE:
                    res.append(
                        Instance(token_type=l_type, un_normalized=l_token, normalized=l_normalized)
                    )
    return res


def load_files(file_paths: List[str], load_func=_load_kaggle_text_norm_file) -> List[Instance]:
    """
    Load given list of text files using the `load_func` function.

    Args:
        file_paths: list of file paths
        load_func: loading function

    Returns: flat list of instances
    """
    res = []
    for file_path in file_paths:
        res.extend(load_func(file_path=file_path))
    return res


def clean_generic(text: str) -> str:
    """
    Cleans text without affecting semiotic classes.

    Args:
        text: string

    Returns: cleaned string
    """
    text = text.strip()
    text = text.lower()
    return text


def evaluate(
    preds: List[str], labels: List[str], input: Optional[List[str]] = None, verbose: bool = True
) -> float:
    """
    Evaluates accuracy given predictions and labels.

    Args:
        preds: predictions
        labels: labels
        input: optional, only needed for verbosity
        verbose: if true prints [input], golden labels and predictions

    Returns accuracy
    """
    acc = 0
    nums = len(preds)
    for i in range(nums):
        pred_norm = clean_generic(preds[i])
        label_norm = clean_generic(labels[i])
        if pred_norm == label_norm:
            acc = acc + 1
        else:
            if input:
                print(f"inpu: {json.dumps(input[i])}")
            print(f"gold: {json.dumps(label_norm)}")
            print(f"pred: {json.dumps(pred_norm)}")
    return acc / nums


def training_data_to_tokens(
    data: List[Instance], category: Optional[str] = None
) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Filters the instance list by category if provided and converts it into a map from token type to list of un_normalized and normalized strings

    Args:
        data: list of instances
        category: optional semiotic class category name

    Returns Dict: token type -> (list of un_normalized strings, list of normalized strings)
    """
    result = defaultdict(lambda: ([], []))
    for instance in data:
        if instance.token_type != EOS_TYPE:
            if category is None or instance.token_type == category:
                result[instance.token_type][0].append(instance.un_normalized)
                result[instance.token_type][1].append(instance.normalized)
    return result


def training_data_to_sentences(data: List[Instance]) -> Tuple[List[str], List[str], List[Set[str]]]:
    """
    Takes instance list, creates list of sentences split by EOS_Token
    Args:
        data: list of instances
    Returns (list of unnormalized sentences, list of normalized sentences, list of sets of categories in a sentence)
    """
    # split data at EOS boundaries
    sentences = []
    sentence = []
    categories = []
    sentence_categories = set()

    for instance in data:
        if instance.token_type == EOS_TYPE:
            sentences.append(sentence)
            sentence = []
            categories.append(sentence_categories)
            sentence_categories = set()
        else:
            sentence.append(instance)
            sentence_categories.update([instance.token_type])
    un_normalized = [
        " ".join([instance.un_normalized for instance in sentence]) for sentence in sentences
    ]
    normalized = [
        " ".join([instance.normalized for instance in sentence]) for sentence in sentences
    ]
    return un_normalized, normalized, categories


def post_process_punctuation(text: str) -> str:
    """
    Normalized quotes and spaces

    Args:
        text: text

    Returns: text with normalized spaces and quotes
    """
    text = (
        text.replace("( ", "(")
        .replace(" )", ")")
        .replace("{ ", "{")
        .replace(" }", "}")
        .replace("[ ", "[")
        .replace(" ]", "]")
        .replace("  ", " ")
        .replace("”", '"')
        .replace("’", "'")
        .replace("»", '"')
        .replace("«", '"')
        .replace("\\", "")
        .replace("„", '"')
        .replace("´", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("‘", "'")
        .replace("`", "'")
        .replace("- -", "--")
    )

    for punct in "!,.:;?":
        text = text.replace(f" {punct}", punct)
    return text.strip()


def pre_process(text: str) -> str:
    """
    Optional text preprocessing before normalization (part of TTS TN pipeline)

    Args:
        text: string that may include semiotic classes

    Returns: text with spaces around punctuation marks
    """
    space_both = "[]"
    for punct in space_both:
        text = text.replace(punct, " " + punct + " ")

    # remove extra space
    text = re.sub(r" +", " ", text)
    return text


def load_file(file_path: str) -> List[str]:
    """
    Loads given text file with separate lines into list of string.

    Args:
        file_path: file path

    Returns: flat list of string
    """
    res = []
    with open(file_path, "r") as fp:
        for line in fp:
            res.append(line)
    return res


def write_file(file_path: str, data: List[str]):
    """
    Writes out list of string to file.

    Args:
        file_path: file path
        data: list of string

    """
    with open(file_path, "w") as fp:
        for line in data:
            fp.write(line + "\n")


def post_process_punct(input: str, normalized_text: str, add_unicode_punct: bool = False):
    """
    Post-processing of the normalized output to match input in terms of spaces around punctuation marks.
    After NN normalization, Moses detokenization puts a space after
    punctuation marks, and attaches an opening quote "'" to the word to the right.
    E.g., input to the TN NN model is "12 test' example",
    after normalization and detokenization -> "twelve test 'example" (the quote is considered to be an opening quote,
    but it doesn't match the input and can cause issues during TTS voice generation.)
    The current function will match the punctuation and spaces of the normalized text with the input sequence.
    "12 test' example" -> "twelve test 'example" -> "twelve test' example" (the quote was shifted to match the input).

    Args:
        input: input text (original input to the NN, before normalization or tokenization)
        normalized_text: output text (output of the TN NN model)
        add_unicode_punct: set to True to handle unicode punctuation marks as well as default string.punctuation (increases post processing time)
    """
    # in the post-processing WFST graph "``" are repalced with '"" quotes (otherwise single quotes "`" won't be handled correctly)
    # this function fixes spaces around them based on input sequence, so here we're making the same double quote replacement
    # to make sure these new double quotes work with this function
    if "``" in input and "``" not in normalized_text:
        input = input.replace("``", '"')
    input = [x for x in input]
    normalized_text = [x for x in normalized_text]
    punct_marks = [x for x in string.punctuation if x in input]

    if add_unicode_punct:
        punct_unicode = [
            chr(i)
            for i in range(sys.maxunicode)
            if category(chr(i)).startswith("P") and chr(i) not in punct_default and chr(i) in input
        ]
        punct_marks = punct_marks.extend(punct_unicode)

    for punct in punct_marks:
        try:
            equal = True
            if input.count(punct) != normalized_text.count(punct):
                equal = False
            idx_in, idx_out = 0, 0
            while punct in input[idx_in:]:
                idx_out = normalized_text.index(punct, idx_out)
                idx_in = input.index(punct, idx_in)

                def _is_valid(idx_out, idx_in, normalized_text, input):
                    """Check if previous or next word match (for cases when punctuation marks are part of
                    semiotic token, i.e. some punctuation can be missing in the normalized text)"""
                    return (
                        idx_out > 0
                        and idx_in > 0
                        and normalized_text[idx_out - 1] == input[idx_in - 1]
                    ) or (
                        idx_out < len(normalized_text) - 1
                        and idx_in < len(input) - 1
                        and normalized_text[idx_out + 1] == input[idx_in + 1]
                    )

                if not equal and not _is_valid(idx_out, idx_in, normalized_text, input):
                    idx_in += 1
                    continue
                if idx_in > 0 and idx_out > 0:
                    if normalized_text[idx_out - 1] == " " and input[idx_in - 1] != " ":
                        normalized_text[idx_out - 1] = ""

                    elif normalized_text[idx_out - 1] != " " and input[idx_in - 1] == " ":
                        normalized_text[idx_out - 1] += " "

                if idx_in < len(input) - 1 and idx_out < len(normalized_text) - 1:
                    if normalized_text[idx_out + 1] == " " and input[idx_in + 1] != " ":
                        normalized_text[idx_out + 1] = ""
                    elif normalized_text[idx_out + 1] != " " and input[idx_in + 1] == " ":
                        normalized_text[idx_out] = normalized_text[idx_out] + " "
                idx_out += 1
                idx_in += 1
        except:
            logging.debug(f"Skipping post-processing of {''.join(normalized_text)} for '{punct}'")

    normalized_text = "".join(normalized_text)
    return re.sub(r" +", " ", normalized_text)
