import sys
import codecs
import zhconv

decode_result = sys.argv[1]
utt2spk_file = sys.argv[2]
merged_result = "/".join(decode_result.split("/")[:-1]) + "/text_merge"

utt2text = {}
utt2spk = {}
spk2texts = {}
spk2text = {}
meeting2text = {}

with codecs.open(decode_result, "r", "utf-8") as f1:
    with codecs.open(utt2spk_file, "r", "utf-8") as f2:
        for line in f1.readlines():
            try:
                line_list = line.strip().split()
                uttid = line_list[0]
                text = "".join(line_list[1:])
            except:
                continue
            utt2text[uttid] = text
        for line in f2.readlines():
            uttid, spkid = line.strip().split()
            utt2spk[uttid] = spkid

for utt, text in utt2text.items():
    spk = utt2spk[utt]
    stime = int(utt.split("-")[-2])
    if spk in spk2texts.keys():
        spk2texts[spk].append([stime, text])
    else:
        spk2texts[spk] = [[stime, text]]

for spk, texts in spk2texts.items():
    texts = sorted(texts, key=lambda x: x[0])
    text = "".join([x[1] for x in texts])
    spk2text[spk] = text

with codecs.open(merged_result, "w", "utf-8") as f:
    for spk, text in spk2text.items():
        # meeting = spk.split("-")[2]
        meeting = spk.split("-")[0]
        if meeting in meeting2text.keys():
            meeting2text[meeting] = meeting2text[meeting] + "$" + text
        else:
            meeting2text[meeting] = text
    for meeting, text in meeting2text.items():
        f.write("%s %s\n" % (meeting, text)) 

