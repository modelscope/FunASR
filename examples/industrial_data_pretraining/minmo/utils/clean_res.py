import re
import sys

in_f = sys.argv[1]
out_f = sys.argv[2]

with open(in_f, "r") as infile, open(out_f, "w") as outfile:
    for line in infile:
        key, response = line.strip().split(maxsplit=1)
        cleaned_response = re.sub(r"[^\w\s\u3000\u4e00-\u9fff]+", "", response)
        outfile.write(key + "\t" + cleaned_response + "\n")
