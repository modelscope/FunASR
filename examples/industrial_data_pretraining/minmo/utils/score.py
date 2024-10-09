import sys
from sklearn.metrics import classification_report


aed_ref = sys.argv[1]
aed_hyp = sys.argv[2]
select_emo = sys.argv[3]
# select_emo = "happy,sad,angry,neutral" #参与打分的情感
emo_list = select_emo.split(",")

ref, hyp = {}, {}
all_key = set()
mix_map = {}

with open(aed_ref, "r") as f:
    for line in f:
        id, event = line.strip().split(" ", 1)[0], line.strip().split(" ", 1)[1]
        ref[id] = event

with open(aed_hyp, "r") as f:
    for line in f:
        if len(line.strip().split(" ", 1)) != 2:
            continue
        id, event = line.strip().split(" ", 1)
        hyp[id] = event


ref_list = []
hyp_list = []


emo_dict = {}


def get_emo(s):
    if "Happy" in s or "开心" in s:
        return "happy"
    if "Sad" in s or "难过" in s:
        return "sad"
    if "Angry" in s or "生气" in s:
        return "angry"
    if "Neutral" in s or "平静" in s:
        return "neutral"
    if "Fearful" in s or "害怕" in s:
        return "fearful"
    if "Surprised" in s or "吃惊" in s:
        return "surprised"
    if "Disgusted" in s or "厌恶" in s:
        return "disgusted"
    return "other"


for key in hyp:
    if key not in ref:
        continue

    ref_emo = get_emo(ref[key])
    hyp_emo = get_emo(hyp[key])

    print(key, ref_emo, hyp_emo)

    if get_emo(ref[key]) not in emo_list or get_emo(hyp[key]) not in select_emo:
        continue

    ref_list.append(get_emo(ref[key]))
    hyp_list.append(get_emo(hyp[key]))

    if ref_emo not in emo_dict:
        emo_dict[ref_emo] = {}
    if hyp_emo not in emo_dict[ref_emo]:
        emo_dict[ref_emo][hyp_emo] = 0
    emo_dict[ref_emo][hyp_emo] += 1


head_line = "*" * 10
hyp_emo_set = set(hyp_list)

for hyp_emo in hyp_emo_set:
    head_line += f"\t{hyp_emo:10}"
print(head_line)
for ref_emo in emo_list:
    if ref_emo not in emo_dict:
        continue
    show_str = [f"{ref_emo:10}"]
    for hyp_emo in hyp_emo_set:
        hyp_num = f"{emo_dict[ref_emo].get(hyp_emo, 0)}"
        show_str.append(f"\t{hyp_num:10}")
    print("".join(show_str))

if len(ref_list) > 0:
    print(classification_report(ref_list, hyp_list, digits=3))

# 使用方法：
# >>> python3 score.py path/to/ref path/to/hyp happy,sad,angry,neutral

# # ref和hyp格式与wav.scp相似: wav_id emotion
# # wav_1 happy
# # wav_2 sad

# 结果示例，
# **********      angry           disgusted       fearful         happy           neutral         sad             surprised
# angry           138             3               1               54              88              20              15
# disgusted       25              2               1               16              16              4               3
# fearful         12              0               1               12              16              6               2
# happy           48              1               0               208             79              18              8
# neutral         147             2               12              298             590             80              17
# sad             41              1               1               32              52              61              9
# surprised       53              1               2               54              85              8               44
# happy: 208/353   recall: 0.589235        acc: 0.351351
# sad: 61/186      recall: 0.327957        acc: 0.340782
# angry: 138/300   recall: 0.460000        acc: 0.368984
# neutral: 590/1115        recall: 0.529148        acc: 0.729295
# UA:0.476585, WA: 0.510235

# ==========
#               precision    recall  f1-score   support

#        angry      0.369     0.460     0.409       300
#        happy      0.351     0.589     0.440       353
#      neutral      0.729     0.529     0.613      1115
#          sad      0.341     0.328     0.334       186

#     accuracy                          0.510      1954
#    macro avg      0.448     0.477     0.449      1954   <--------以这两行为准
# weighted avg      0.569     0.510     0.524      1954   <--------以这两行为准
