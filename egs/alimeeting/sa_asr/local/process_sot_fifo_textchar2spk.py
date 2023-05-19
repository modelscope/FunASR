# -*- coding: utf-8 -*-
"""
Process the textgrid files
"""
import argparse
import codecs
from distutils.util import strtobool
from pathlib import Path
import textgrid
import pdb

def get_args():
    parser = argparse.ArgumentParser(description="process the textgrid files")
    parser.add_argument("--path", type=str, required=True, help="Data path")
    args = parser.parse_args()
    return args

class Segment(object):
    def __init__(self, uttid, text):
        self.uttid = uttid
        self.text = text

def main(args):
    text = codecs.open(Path(args.path) / "text", "r", "utf-8")
    spk2utt = codecs.open(Path(args.path) / "spk2utt", "r", "utf-8")
    utt2spk = codecs.open(Path(args.path) / "utt2spk_all_fifo", "r", "utf-8")   
    spk2id = codecs.open(Path(args.path) / "spk2id", "w", "utf-8")
    
    spkid_map = {}
    meetingid_map = {}
    for line in spk2utt:
        spkid = line.strip().split(" ")[0]
        meeting_id_list = spkid.split("_")[:3]
        meeting_id = meeting_id_list[0] + "_" + meeting_id_list[1] + "_" + meeting_id_list[2]
        if meeting_id not in meetingid_map:
            meetingid_map[meeting_id] = 1     
        else:
            meetingid_map[meeting_id] += 1
        spkid_map[spkid] = meetingid_map[meeting_id]
        spk2id.write("%s %s\n" % (spkid, meetingid_map[meeting_id]))
    
    utt2spklist = {}
    for line in utt2spk:
        uttid = line.strip().split(" ")[0]
        spkid = line.strip().split(" ")[1]
        spklist = spkid.split("$")
        tmp = []
        for index in range(len(spklist)):
            tmp.append(spkid_map[spklist[index]])
        utt2spklist[uttid] = tmp
    # parse the textgrid file for each utterance
    all_segments = []
    for line in text:
        uttid = line.strip().split(" ")[0]
        context = line.strip().split(" ")[1]
        spklist = utt2spklist[uttid]
        length_text = len(context)
        cnt = 0
        tmp_text = ""
        for index in range(length_text):
            if context[index] != "$":
                tmp_text += str(spklist[cnt])
            else:
                tmp_text += "$"
                cnt += 1
        tmp_seg = Segment(uttid,tmp_text)
        all_segments.append(tmp_seg)

    text.close()
    utt2spk.close()
    spk2utt.close()
    spk2id.close()

    text_id = codecs.open(Path(args.path) / "text_id", "w", "utf-8")

    for i in range(len(all_segments)):
        uttid_tmp = all_segments[i].uttid
        text_tmp = all_segments[i].text
        
        text_id.write("%s %s\n" % (uttid_tmp, text_tmp))

    text_id.close()

if __name__ == "__main__":
    args = get_args()
    main(args)
