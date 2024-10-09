import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file) as f:
    lines = f.readlines()

with open(output_file, "w") as wf:
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            key, text = parts
        else:
            key, parts = parts[0], " "
        text = [t for t in text.replace(" ", "")]
        text = " ".join(text)
        wf.write(key + " " + text + "\n")
