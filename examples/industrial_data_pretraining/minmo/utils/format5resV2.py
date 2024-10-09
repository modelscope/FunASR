# -*- coding: utf-8 -*-
#!/usr/bin/python
import sys, re

# f=sys.stdin


def scoreformat(name, line, flag=0):
    newline = ""
    for i in range(0, len(line)):
        curr = line[i]
        currEn = False
        if curr == "":
            continue
        if curr.upper() >= "A" and curr.upper() <= "Z" or curr == "'":
            currEn = True
        if i == 0:
            newline = newline + curr.upper()
        else:
            if lastEn == True and currEn == True:
                newline = newline + curr.upper()
            else:
                newline = newline + " " + curr.upper()
        if flag == -1:
            lastEn = False
        else:
            lastEn = currEn
    ret = re.sub("[ ]{1,}", " ", newline)
    ret = ret
    if flag <= 0:
        ret = ret + " " + "(" + name + ")"
    else:
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
        #    print line[i],ord(word)
        if (word >= "\u4e00" and word <= "\u9fa5") or (word >= "\u0030" and word <= "\u0039"):
            if en_flag == 1:
                newline += " " + word
            else:
                newline += word
            en_flag = 0
            # print "-----",newline
        elif (
            (word >= "\u0041" and word <= "\u005a")
            or (word >= "\u0061" and word <= "\u007a")
            or word == "'"
        ):
            if en_flag == 0:
                newline += " " + ("" if (word == "'") else word)
            else:
                newline += word
            en_flag = 1
            # print "+++",newline
        else:
            newline += " " + word
            # print "0-0-0-0",newline
    newline = newline
    newline = re.sub("[ ]{1,}", " ", newline)
    newline = newline
    return newline


def numbersingle(line):
    chnu = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "点"]
    newline = ""
    for id in range(len(line)):
        if re.findall(r"\.", line[id]):
            if re.findall(r"\.\s*$", line[id]):
                newline += "."
            else:
                newline += chnu[10]
        elif re.search(r"0", line[id]):
            if id > 0 and id < len(line) - 1:
                if (
                    re.search(r"\d", line[id - 1])
                    and (not re.search(r"\d", line[id + 1]))
                    and (not re.search(r"0", line[id - 1]))
                ):
                    if id > 2 and len(line) > 2 and (not re.search(r"\d", line[id - 1])):
                        newline = newline[:-1]
                        newline += chnu[int(line[id - 1])] + "十"
                    else:
                        newline += chnu[int(line[id])]
                else:
                    newline += chnu[int(line[id])]
            else:
                newline += chnu[int(line[id])]
        elif re.search(r"\d", line[id]):
            newline += chnu[int(line[id])]
        else:
            newline += line[id]
    return newline


def ch_number2digit(line):
    number_flag = 0
    zero_flag = 0
    # print "ch_umber2digit---------",line
    bits = {
        "零": "1",
        "十": "2",
        "百": "3",
        "千": "4",
        "万": "5",
        "十万": "6",
        "百万": "7",
        "千万": "8",
    }
    # chnu={'零':"0",'一':"1",'二':"2",'三':"3",'四':"4",'五':"5",'六':"6",'七':"7",'八':"8",'九':"9",'十':"10"]
    chsh = {
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "两": "2",
        "幺": "1",
    }
    unit = {"里": "1", "克": "1", "米": "1"}
    newline = ""
    digit = []
    bit = []
    onebit = ""
    #  digitstr=""
    for i in range(len(line)):
        if ord(line[i]) == 32:
            newline += " "
            continue
        #    print line[i],str(ord(line[i]))
        if line[i] in chsh:
            number_flag = 1
            if line[i] == "两":
                if (i == len(line) - 1) or (
                    (line[i + 1] not in chsh.keys()) and (line[i + 1] not in bits.keys())
                ):
                    number_flag = -1
            if number_flag == 1:
                digit.append(chsh[line[i]])

        elif "十" == line[i] and number_flag == 0:
            number_flag = 2
            digit.append("1")
            bit.append(line[i])
        elif "十" == line[i] and number_flag == 3:
            digit.append("1")
            bit.append(line[i])
        elif ("零" == line[i]) and (number_flag == 0 or number_flag == 1):
            digit.append("0")
        elif ("零" == line[i]) and number_flag == 3:
            zero_flag = 1
        elif number_flag == 1 and line[i] in bits:
            number_flag = 3
            if line[i] == "千":
                if i < len(line) - 1:
                    if line[i + 1] in unit:
                        number_flag = -1
            if number_flag == 3:
                onebit = line[i]
                bit.append(onebit)
        elif number_flag == 3 and line[i] in bits:
            onebit = bit[-1] + line[i]
            if onebit in bits:
                bit[-1] = onebit
            else:
                number_flag = -2
        else:
            number_flag = -1
        if len(digit) > 0 and number_flag == -1:
            number_flag = -2
        if i == (len(line) - 1) and number_flag >= 0:
            number_flag = -1
        #    print "number_end_flag",number_flag
        if number_flag < 0:
            newdigit = ""
            # print digit
            # print "length:",len(bit), #bit[0]#,bit[0]
            if len(digit) > 0:  # and (len(digit) == len(bit))):
                if len(bit) == 1 and zero_flag == 0 and bit[0] == "百" and len(bit) != len(digit):
                    bit.append("十")
                if len(digit) == (len(bit) + 1):
                    bit.append("零")
                #        print digit[:]
                #        print bit[:]
                if len(digit) == len(bit):
                    for m in range(len(digit))[-1::-1]:
                        if int(bits[bit[m]]) == int(len(newdigit) + 1):
                            newdigit += digit[m]
                        else:
                            nu = int(bits[bit[m]]) - len(newdigit) - 1
                            for n in range(nu):
                                newdigit += "0"
                            newdigit += digit[m]
                    for z in range(len(newdigit))[-1::-1]:
                        newline += newdigit[z]
                else:
                    newline += "".join(digit)
                bit = []
                digit = []
                zero_flag = 0
            else:
                newline += line[i]
            if number_flag == -2:
                newline += line[i]
            number_flag = 0
    return newline


def special(line):
    # 	print line
    newline = ""
    for e in range(len(line)):
        # print "e ord\t",line[e],ord(line[e])
        if ord(line[e]) == 247:
            newline += "除以"
        elif ord(line[e]) == 215:
            newline += "乘以"
        elif ord(line[e]) == 61:
            newline += "等于"
        elif ord(line[e]) == 43:
            newline += "加"
        elif ord(line[e]) == 45:
            newline += "负"
        elif ord(line[e]) == 8451:
            newline += "摄氏度"
        elif ord(line[e]) == 13217:
            newline += "平方米"
        elif ord(line[e]) == 8240 or ord(line[e]) == 65130:
            newline += "%"
        elif ord(line[e]) == 46:
            newline += "点"
        elif ord(line[e]) == 176:
            newline += "度"
            angel = 1
        elif ord(line[e]) == 8242 and angel == 1:
            newline += "分"
        else:
            newline += line[e]
    return newline


if __name__ == "__main__":
    if len(sys.argv[1:]) < 1:
        sys.stderr.write("Usage:\n .py  reco.result\n")
        sys.stderr.write(" reco.result:   id<tab>recoresult\n")
        sys.exit(1)
    f = open(sys.argv[1])
    flag = 0
    if len(sys.argv[1:]) > 1:
        flag = int(sys.argv[2])
    for line in f.readlines():
        if not line:
            continue
        line = line.rstrip()
        #    print line
        tmp = line.split("\t")
        if len(tmp) < 2:
            tmp = line.split(",")
            if len(tmp) < 2:
                tmp = line.split(" ", 1)
                if len(tmp) < 2:
                    name = tmp[0]
                    content = ""
                    print(content)
                    continue
        name = tmp[0]
        content = tmp[1]
        name = re.sub("\.pcm", "", name)
        name = re.sub("\.wav", "", name)
        content = recoformat(content)
        content = numbersingle(content)
        #    print "single",content
        content = ch_number2digit(content)
        # print "digit",content
        content = special(content)
        #  print "special",content
        content = scoreformat(name, content, flag)
        print(content)
    f.close()
