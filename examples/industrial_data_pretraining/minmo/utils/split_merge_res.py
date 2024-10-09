import sys


input_file = sys.argv[1]
key_file = sys.argv[2]
output_file = sys.argv[3]


key_dct = {}
with open(key_file, "r") as f:
    for line in f:
        content = line.strip().split(" ", 1)
        if len(content) == 2:
            key, trans = content[0], content[1]
        else:
            key = content[0]
            trans = ""
        key_dct[key] = trans

fout = open(output_file, "w")

repeat_lst = []

with open(input_file, "r") as f:
    for line in f:
        content = line.strip().split(" ", 1)
        if len(content) == 2:
            key, trans = content[0], content[1]
        else:
            key = content[0]
            trans = ""
        if key in key_dct and key not in repeat_lst:
            repeat_lst.append(key)
            fout.writelines(key + " " + trans + "\n")
