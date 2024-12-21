import re
import argparse


def load_dict(seg_file):
    seg_dict = {}
    with open(seg_file, "r") as infile:
        for line in infile:
            s = line.strip().split()
            key = s[0]
            value = s[1:]
            seg_dict[key] = " ".join(value)
    return seg_dict


def forward_segment(text, dic):
    word_list = []
    i = 0
    while i < len(text):
        longest_word = text[i]
        for j in range(i + 1, len(text) + 1):
            word = text[i:j]
            if word in dic:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.append(longest_word)
        i += len(longest_word)
    return word_list


def tokenize(txt, seg_dict):
    out_txt = ""
    pattern = re.compile(r"([\u4E00-\u9FA5A-Za-z0-9])")
    for word in txt:
        if pattern.match(word):
            if word in seg_dict:
                out_txt += seg_dict[word] + " "
            else:
                out_txt += "<unk>" + " "
        else:
            continue
    return out_txt.strip()


def get_parser():
    parser = argparse.ArgumentParser(
        description="text tokenize",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--text-file",
        "-t",
        default=False,
        required=True,
        type=str,
        help="input text",
    )
    parser.add_argument(
        "--seg-file",
        "-s",
        default=False,
        required=True,
        type=str,
        help="seg file",
    )
    parser.add_argument(
        "--txt-index",
        "-i",
        default=1,
        required=True,
        type=int,
        help="txt index",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=False,
        required=True,
        type=str,
        help="output dir",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    txt_writer = open("{}/text.{}.txt".format(args.output_dir, args.txt_index), "w")
    shape_writer = open("{}/len.{}".format(args.output_dir, args.txt_index), "w")
    seg_dict = load_dict(args.seg_file)
    with open(args.text_file, "r") as infile:
        for line in infile:
            s = line.strip().split()
            text_id = s[0]
            text_list = forward_segment("".join(s[1:]).lower(), seg_dict)
            text = tokenize(text_list, seg_dict)
            lens = len(text.strip().split())
            txt_writer.write(text_id + " " + text + "\n")
            shape_writer.write(text_id + " " + str(lens) + "\n")


if __name__ == "__main__":
    main()
