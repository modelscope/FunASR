import argparse
import tqdm
import codecs
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


def main(args):
    tg = textgrid.TextGrid.fromFile(args.input_textgrid_file)
    segments = []
    spk = {}
    num_spk = 1
    uttid = args.uttid
    for i in range(tg.__len__()):
        for j in range(tg[i].__len__()):
            if tg[i][j].mark:
                if tg[i].name not in spk:
                    spk[tg[i].name] = num_spk
                    num_spk += 1
                segments.append(
                    Segment(
                        uttid,
                        spk[tg[i].name],
                        tg[i][j].minTime,
                        tg[i][j].maxTime,
                        tg[i][j].mark.strip(),
                    )
                )
    segments = sorted(segments, key=lambda x: x.stime)

    rttm_file = codecs.open(args.output_rttm_file, "w", "utf-8")

    for i in range(len(segments)):
        fmt = "SPEAKER {:s} 1 {:.2f} {:.2f} <NA> <NA> {:s} <NA> <NA>"
        #pdb.set_trace()
        rttm_file.write(f"{fmt.format(segments[i].uttid, float(segments[i].stime), float(segments[i].etime) - float(segments[i].stime), str(segments[i].spkr))}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make rttm for true label")
    parser.add_argument("--input_textgrid_file", required=True, help="The textgrid file")
    parser.add_argument("--output_rttm_file", required=True, help="The output rttm file")
    parser.add_argument("--uttid", required=True, help="The utt id of the file")
    args = parser.parse_args()
    main(args)
