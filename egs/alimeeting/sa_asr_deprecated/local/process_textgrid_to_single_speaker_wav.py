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
import numpy as np
import sys
import math


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
    parser.add_argument("--path", type=str, required=True, help="Data path")
    args = parser.parse_args()
    return args



def main(args):
    textgrid_flist = codecs.open(Path(args.path) / "textgrid.flist", "r", "utf-8")
    segment_file = codecs.open(Path(args.path)/"segments", "w", "utf-8")
    utt2spk = codecs.open(Path(args.path)/"utt2spk", "w", "utf-8")

    # get the path of textgrid file for each utterance
    for line in textgrid_flist:
        line_array = line.strip().split(" ")
        path = Path(line_array[1])
        uttid = line_array[0]

        try:
            tg = textgrid.TextGrid.fromFile(path)
        except:
            pdb.set_trace()
        num_spk = tg.__len__()
        spk2textgrid = {}
        spk2weight = {}
        weight2spk = {}
        cnt = 2
        xmax = 0
        for i in range(tg.__len__()):
            spk_name = tg[i].name
            if spk_name not in spk2weight:
                spk2weight[spk_name] = cnt
                weight2spk[cnt] = spk_name
                cnt = cnt * 2
            segments = []
            for j in range(tg[i].__len__()):
                if tg[i][j].mark:
                    if xmax < tg[i][j].maxTime:
                        xmax = tg[i][j].maxTime
                    segments.append(
                        Segment(
                            uttid,
                            tg[i].name,
                            tg[i][j].minTime,
                            tg[i][j].maxTime,
                            tg[i][j].mark.strip(),
                        )
                    )
            segments = sorted(segments, key=lambda x: x.stime)
            spk2textgrid[spk_name] = segments
        olp_label = np.zeros((num_spk, int(xmax/0.01)), dtype=np.int32)
        for spkid in spk2weight.keys():
            weight = spk2weight[spkid]
            segments = spk2textgrid[spkid]
            idx = int(math.log2(weight) )- 1
            for i in range(len(segments)):
                stime = segments[i].stime
                etime = segments[i].etime
                olp_label[idx, int(stime/0.01): int(etime/0.01)] = weight
        sum_label = olp_label.sum(axis=0)
        stime = 0
        pre_value = 0
        for pos in range(sum_label.shape[0]):
            if sum_label[pos] in weight2spk:
                if pre_value in weight2spk:
                    if sum_label[pos] != pre_value:    
                        spkids = weight2spk[pre_value]
                        spkid_array = spkids.split("_")
                        spkid = spkid_array[-1]
                        #spkid = uttid+spkid 
                        if round(stime*0.01, 2) != round((pos-1)*0.01, 2):
                            segment_file.write("%s_%s_%s_%s %s %s %s\n" % (uttid, spkid, str(int(stime)).zfill(7), str(int(pos-1)).zfill(7), uttid, round(stime*0.01, 2) ,round((pos-1)*0.01, 2)))
                            utt2spk.write("%s_%s_%s_%s %s\n" % (uttid, spkid, str(int(stime)).zfill(7), str(int(pos-1)).zfill(7), uttid+"_"+spkid))
                        stime = pos
                        pre_value = sum_label[pos]
                else:
                    stime = pos
                    pre_value = sum_label[pos]
            else:
                if pre_value in weight2spk:
                    spkids = weight2spk[pre_value]
                    spkid_array = spkids.split("_")
                    spkid = spkid_array[-1]
                    #spkid = uttid+spkid 
                    if round(stime*0.01, 2) != round((pos-1)*0.01, 2):
                        segment_file.write("%s_%s_%s_%s %s %s %s\n" % (uttid, spkid, str(int(stime)).zfill(7), str(int(pos-1)).zfill(7), uttid, round(stime*0.01, 2) ,round((pos-1)*0.01, 2)))
                        utt2spk.write("%s_%s_%s_%s %s\n" % (uttid, spkid, str(int(stime)).zfill(7), str(int(pos-1)).zfill(7), uttid+"_"+spkid))
                    stime = pos
                    pre_value = sum_label[pos]
    textgrid_flist.close()
    segment_file.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
