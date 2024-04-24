#!/usr/bin/env python

# Apache 2.0

import sys

with open(sys.argv[1], "r") as fread:
    print("0 0 <blank> <eps>")
    nodeX = 1
    for entry in fread.readlines():
        entry = entry.replace("\n", "").strip()
        fields = entry.split(" ")
        phone = fields[0]
        if phone == "<eps>" or phone == "<blank>":
            continue
        if "#" in phone:
            print(str(0) + " " + str(0) + " " + "<eps>" + " " + phone)
        else:
            print(str(0) + " " + str(nodeX) + " " + phone + " " + phone)
            print(str(nodeX) + " " + str(nodeX) + " " + phone + " <eps>")
            print(str(nodeX) + " " + str(0) + " " + "<eps> <eps>")
        nodeX += 1
    print("0")
