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

class Segment(object):
    def __init__(self, uttid, spkr, stime, etime, text):
        self.uttid = uttid
        self.spkr = spkr
        self.stime = round(stime, 2)
        self.etime = round(etime, 2)
        self.text = text

    def change_stime(self, time):
        self.stime = time

    def change_etime(self, time):
        self.etime = time


def get_args():
    parser = argparse.ArgumentParser(description="process the textgrid files")
    parser.add_argument("--path", type=str, required=True, help="textgrid path")
    parser.add_argument("--label_path", type=str, required=True, help="label rttm file path")
    parser.add_argument("--predict_path", type=str, required=True, help="predict rttm file path")
    args = parser.parse_args()
    return args

def main(args):
    textgrid_flist = codecs.open(Path(args.path)/"uttid_textgrid.flist", "r", "utf-8")


    # parse the textgrid file for each utterance
    speaker2_uttidset = []
    speaker3_uttidset = []
    speaker4_uttidset = []
    for line in textgrid_flist:
        uttid ,textgrid_file = line.strip().split("\t")
        tg = textgrid.TextGrid()
        tg.read(textgrid_file)
        
        num_speaker = len(tg)
        if num_speaker ==2:
            speaker2_uttidset.append(uttid)
        elif num_speaker ==3:
            speaker3_uttidset.append(uttid)
        elif num_speaker ==4:
            speaker4_uttidset.append(uttid)
    textgrid_flist.close()

    speaker2_id_label = codecs.open(Path(args.label_path) / "speaker2_id", "w", "utf-8")
    speaker2_id_predict = codecs.open(Path(args.predict_path) / "speaker2_id", "w", "utf-8")
    speaker3_id_label = codecs.open(Path(args.label_path) / "speaker3_id", "w", "utf-8")
    speaker3_id_predict = codecs.open(Path(args.predict_path) / "speaker3_id", "w", "utf-8")
    speaker4_id_label = codecs.open(Path(args.label_path) / "speaker4_id", "w", "utf-8")
    speaker4_id_predict = codecs.open(Path(args.predict_path) / "speaker4_id", "w", "utf-8")

    for i in range(len(speaker2_uttidset)):
        speaker2_id_label.write("%s\n" % (args.label_path+"/"+speaker2_uttidset[i]+".rttm"))
        speaker2_id_predict.write("%s\n" % (args.predict_path+"/"+speaker2_uttidset[i]+".rttm"))
    for i in range(len(speaker3_uttidset)):
        speaker3_id_label.write("%s\n" % (args.label_path+"/"+speaker3_uttidset[i]+".rttm"))
        speaker3_id_predict.write("%s\n" % (args.predict_path+"/"+speaker3_uttidset[i]+".rttm"))
    for i in range(len(speaker4_uttidset)):
        speaker4_id_label.write("%s\n" % (args.label_path+"/"+speaker4_uttidset[i]+".rttm"))
        speaker4_id_predict.write("%s\n" % (args.predict_path+"/"+speaker4_uttidset[i]+".rttm"))

    speaker2_id_label.close()
    speaker2_id_predict.close()
    speaker3_id_label.close()
    speaker3_id_predict.close()
    speaker4_id_label.close()
    speaker4_id_predict.close()

if __name__ == "__main__":
    args = get_args()
    main(args)
