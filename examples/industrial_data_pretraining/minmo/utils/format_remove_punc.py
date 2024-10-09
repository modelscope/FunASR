import sys
import string

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        content = line.strip().split("\t", 1)
        if len(content) == 2:
            utt, text = content[0], content[1]
        else:
            utt = content[0]
            text = ""
        # 创建一个翻译表，将所有标点符号（除了撇号）映射为 None
        translator = str.maketrans("", "", string.punctuation.replace("'", ""))

        # 使用翻译表去除标点符号
        no_punctuation_text = text.translate(translator)

        # 将所有英文字符转换成小写
        lowercase_text = no_punctuation_text.lower()

        outfile.write(utt + "\t" + lowercase_text + "\n")
