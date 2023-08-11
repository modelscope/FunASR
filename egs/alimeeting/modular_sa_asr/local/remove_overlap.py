import sys
import pdb
import codecs
import os

input_segments_file = sys.argv[1]
input_utt2spk_file = sys.argv[2]
output_segments_file = sys.argv[3]
output_utt2spk_file = sys.argv[4]
threshold = sys.argv[5]


class Segment(object):
    def __init__(self, baseid, spkid, meetingid, stime, etime, uttid=None):
        self.baseid = baseid
        self.spkid = spkid
        self.meetingid = meetingid
        self.stime = round(stime, 2)
        self.etime = round(etime, 2)
        self.uttid = uttid
        self.dur = self.etime - self.stime
        if self.uttid is None:
            self.uttid = "%s-%s-%07d-%07d" % (
                self.baseid,
                self.spkid,
                self.stime * 100,
                self.etime * 100,
            )


def cut(cur_max_end_time, seg_list, cur_seg, next_c):
    global out_segment_dict
    if next_c == len(seg_list):
        single_stime = max(cur_max_end_time, cur_seg.stime)
        single_etime = cur_seg.etime
        if single_stime < single_etime and single_etime - single_stime > float(threshold):
            # only save segment which duration more than threshold for sv's accuracy
            if cur_seg.spkid not in out_segment_dict.keys():
                out_segment_dict[cur_seg.spkid] = [
                    Segment(
                        cur_seg.baseid,
                        cur_seg.spkid,
                        cur_seg.meetingid,
                        single_stime,
                        single_etime,
                    )]
            else:
                out_segment_dict[cur_seg.spkid].append(
                    Segment(
                        cur_seg.baseid,
                        cur_seg.spkid,
                        cur_seg.meetingid,
                        single_stime,
                        single_etime,
                    )
                )
    else:
        next_seg = seg_list[next_c]
        single_stime = max(cur_max_end_time, cur_seg.stime)
        single_etime = min(cur_seg.etime, next_seg.stime)
        if single_stime < single_etime and single_etime - single_stime > float(threshold):
            if cur_seg.spkid not in out_segment_dict.keys():
                out_segment_dict[cur_seg.spkid] = [
                    Segment(
                        cur_seg.baseid,
                        cur_seg.spkid,
                        cur_seg.meetingid,
                        single_stime,
                        single_etime,
                )]
            else:
                out_segment_dict[cur_seg.spkid].append(
                    Segment(
                        cur_seg.baseid,
                        cur_seg.spkid,
                        cur_seg.meetingid,
                        single_stime,
                        single_etime,
                    )
                )
        if cur_seg.etime > next_seg.etime:
            cut(max(cur_max_end_time, next_seg.etime), seg_list, cur_seg, next_c + 1)


meeting2seg = {}
utt2spk = {}
i = 0

with codecs.open(input_utt2spk_file, "r", "utf-8") as f:
    for line in f.readlines():
        utt, spk = line.strip().split()
        utt2spk[utt] = spk

with codecs.open(input_segments_file, "r", "utf-8") as f:
    for line in f.readlines():
        i += 1
        uttid, meetingid, stime, etime = line.strip().split(" ")
        spkid = utt2spk[uttid].split("-")[1]
        baseid = meetingid
        one_seg = Segment(baseid, spkid, meetingid, float(stime), float(etime))
        if one_seg.meetingid not in meeting2seg.keys():
            meeting2seg[one_seg.meetingid] = [one_seg]
        else:
            meeting2seg[one_seg.meetingid].append(one_seg)

out_segment_dict = {}

for k, v in meeting2seg.items():
    meeting2seg[k] = sorted(v, key=lambda x: x.stime)
    cur_max_end_time = 0
    for i in range(len(v)):
        cur_seg = meeting2seg[k][i]
        cut(cur_max_end_time, meeting2seg[k], cur_seg, i + 1)
        cur_max_end_time = max(cur_max_end_time, cur_seg.etime)

out_segment_list = []

for k, v in out_segment_dict.items():
    out_segment_list.extend(out_segment_dict[k])

with codecs.open(output_segments_file, "w", "utf-8") as f_seg:
    with codecs.open(output_utt2spk_file, "w", "utf-8") as f_utt2spk:
        for out_seg in out_segment_list:
            f_seg.write(
                "%s %s %.2f %.2f\n"
                % (
                    out_seg.uttid,
                    out_seg.meetingid,
                    out_seg.stime,
                    out_seg.etime,
                )
            )
            f_utt2spk.write(
                "%s %s-%s\n"
                % (
                    out_seg.uttid,
                    out_seg.baseid,
                    out_seg.spkid,
                )
            )