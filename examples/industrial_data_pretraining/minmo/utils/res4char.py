# -*- coding: utf-8 -*-
#!/usr/bin/python
# Author: weijuan
import sys, re


def scoreformat(name, line):
    newline = ""
    for i in range(0, len(line)):
        curr = line[i]
        currEn = False
        if curr == "":
            continue
        if curr.upper() >= "A" and curr.upper() <= "Z" or curr == "'":
            currEn = True
        if i == 0:
            newline = newline + curr.lower()
        else:
            if lastEn == True and currEn == True:
                newline = newline + curr.lower()
            else:
                newline = newline + " " + curr.lower()
        lastEn = currEn
    ret = re.sub("[ ]{1,}", " ", newline)
    ret = ret
    ret = name + "\t" + ret
    return ret


def recoformat(line):
    newline = ""
    en_flag = 0  # 0: no-english   1 : english   2: former
    for i in range(0, len(line)):
        word = line[i]
        if ord(word) == 32:
            if en_flag == 0:
                continue
            else:
                en_flag = 0
                newline += " "
        if (word >= "\u4e00" and word <= "\u9fa5") or (word >= "\u0030" and word <= "\u0039"):
            if en_flag == 1:
                newline += " " + word
            else:
                newline += word
            en_flag = 0
        elif (
            (word >= "\u0041" and word <= "\u005a")
            or (word >= "\u0061" and word <= "\u007a")
            or word == "'"
        ):
            if en_flag == 0:
                newline += " " + word
            else:
                newline += word
            en_flag = 1
        else:
            newline += " "
    newline = newline
    newline = re.sub("[ ]{1,}", " ", newline)
    newline = newline.strip()
    newline = newline
    return newline


if __name__ == "__main__":
    if len(sys.argv[1:]) < 1:
        sys.stderr.write("Usage:\n .py  reco.result\n")
        sys.stderr.write(" reco.result:   id<delimiter>recoresult; delimiter: \\t , blank \n")
        sys.exit(1)
    f = open(sys.argv[1])
    for line in f.readlines():
        if not line:
            continue
        line = line.rstrip()
        if "\t" in line:
            tmp = line.split("\t")
        elif "," in line:
            tmp = line.split(",", 1)
        else:
            tmp = line.split(" ", 1)
        if len(tmp) < 2:
            continue
        name = tmp[0]
        content = tmp[1]
        name = re.sub("\.pcm$", "", name)
        name = re.sub("\.wav$", "", name)
        # name=re.sub("wav[0-9]{3,}_[0-9]{4,}_","",name)
        content = recoformat(content)
        content = scoreformat(name, content)
        print(content)
    f.close()
