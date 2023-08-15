import codecs
import sys

rttm_file_path = sys.argv[1]
segment_file_path = sys.argv[2]
mode = sys.argv[3] # 0 for diarization, 1 for asr


meeting2spk = {}

with codecs.open(rttm_file_path, "r", "utf-8") as fi:
    with codecs.open(segment_file_path + "/segments", "w", "utf-8") as f1:
        with codecs.open(segment_file_path + "/utt2spk", "w", "utf-8") as f2:
            for line in fi.readlines():
                _, sessionid, _, stime, dur, _, _, spkid, _, _ = line.strip().split(" ")
                if float(dur) < 0.3:
                    continue
                uttid = "%s-%07d-%07d" % (sessionid, int(float(stime) * 100), int(float(stime) * 100 + float(dur) * 100))
                spkid = "%s-%s" % (sessionid, spkid)
                if int(mode) == 0:
                    f1.write("%s %s %.2f %.2f\n" % (uttid, sessionid, float(stime), float(stime) + float(dur)))
                    f2.write("%s %s\n" % (uttid, spkid))
                elif int(mode) == 1:
                    f1.write("%s %s %.2f %.2f\n" % (uttid, spkid, float(stime), float(stime) + float(dur)))
                    f2.write("%s %s\n" % (uttid, spkid))
                else:
                    exit("mode only support 0 or 1!")


