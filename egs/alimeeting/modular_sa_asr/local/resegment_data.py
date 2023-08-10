import soundfile
import os
import sys
import codecs
import numpy as np
import pdb


segment_file_path = sys.argv[1]
wav_scp_file_path = sys.argv[2]
data_path = sys.argv[3]

wav_save_path = data_path + "/wav/"
os.system("mkdir -p " + wav_save_path)
pos_path = data_path + "/pos_map/"
os.system("mkdir -p " + pos_path)

wav_dict = {}
seg2time = {}
seg2time_new = {}
session2profile = {}

with codecs.open(wav_scp_file_path, "r", "utf-8") as f:
    for line in f.readlines():
        sessionid, wav_path = line.strip().split()
        wav_dict[sessionid] = wav_path

with codecs.open(segment_file_path, "r", "utf-8") as f:
    for line in f.readlines():
        _, sessionid, stime, etime = line.strip().split()
        if sessionid not in seg2time.keys():
            seg2time[sessionid] = [(int(16000 * float(stime)), int(16000 * float(etime)))]
        else:
            seg2time[sessionid].append((int(16000 * float(stime)), int(16000 * float(etime))))
with codecs.open(data_path + "/map.scp", "w", "utf-8") as f1:
    for sessionid, seg_times in seg2time.items():
        seg2time_new[sessionid] = []
        last_time = 0
        with codecs.open(pos_path + sessionid + ".pos", "w", "utf-8") as f2:
            for seg_time in seg_times:
                tmp = seg_time[0] - last_time
                cur_seg = (seg_time[0] - tmp, seg_time[1] - tmp)
                seg2time_new[sessionid].append((seg_time[0] - last_time, seg_time[1] - last_time))
                last_time = cur_seg[1]
                f2.write("%s-%07d-%07d %d %d %d %d\n" % (sessionid, seg_time[0]/160, seg_time[1]/160, seg_time[0], seg_time[1], cur_seg[0], cur_seg[1]))
        f1.write("%s %s\n" % (sessionid, pos_path + sessionid + ".pos"))

with codecs.open(data_path + "/cluster_profile_zeropadding16.scp", "r", "utf-8") as f:
    for line in f.readlines():
        session, path = line.strip().split()
        session2profile[session] = path

with codecs.open(data_path + "/wav.scp", "w", "utf-8") as f1:
    with codecs.open(data_path + "/profile.scp", "w", "utf-8") as f2:
        for sessionid, wav_path in wav_dict.items():
            wav = soundfile.read(wav_path)[0]
            if wav.ndim == 2:
                    wav = wav[:, 0]
            seg_list = [wav[seg[0]: seg[1]] for seg in seg2time[sessionid]]
            wav_new = np.concatenate(seg_list, axis=0)
            cur_time = 0
            flag = True
            while flag:
                start = cur_time
                end = cur_time + 256000
                if end < wav_new.shape[0]:
                    cur_wav = wav_new[start: end]
                else:
                    cur_wav = wav_new[start: ]
                    flag = False
                cur_time = cur_time + 64000
                wav_name = "%s-%07d_%07d.wav" % (sessionid, start/160, end/160)
                soundfile.write(wav_save_path + wav_name, cur_wav, 16000)
                f1.write("%s %s\n" % (wav_name, wav_save_path + wav_name))
                f2.write("%s %s\n" % (wav_name, session2profile[sessionid]))
                


