import editdistance
import sys
import os
from itertools import permutations


def load_transcripts(file_path):
    trans_list = []
    for one_line in open(file_path, "rt"):
        meeting_id, trans = one_line.strip().split(" ")
        trans_list.append((meeting_id.strip(), trans.strip()))

    return trans_list

def calc_spk_trans(trans):
    spk_trans_ = [x.strip() for x in trans.split("$")]
    spk_trans = []
    for i in range(len(spk_trans_)):
        spk_trans.append((str(i), spk_trans_[i]))
    return spk_trans

def calc_cer(ref_trans, hyp_trans):
    ref_spk_trans = calc_spk_trans(ref_trans)
    hyp_spk_trans = calc_spk_trans(hyp_trans)
    ref_spk_num, hyp_spk_num = len(ref_spk_trans), len(hyp_spk_trans)
    num_spk = max(len(ref_spk_trans), len(hyp_spk_trans))
    ref_spk_trans.extend([("", "")] * (num_spk - len(ref_spk_trans)))
    hyp_spk_trans.extend([("", "")] * (num_spk - len(hyp_spk_trans)))

    errors, counts, permutes = [], [], []
    min_error = 0
    cost_dict = {}
    for perm in permutations(range(num_spk)):
        flag = True
        p_err, p_count = 0, 0
        for idx, p in enumerate(perm):
            if abs(len(ref_spk_trans[idx][1]) - len(hyp_spk_trans[p][1])) > min_error > 0:
                flag = False
                break
            cost_key = "{}-{}".format(idx, p)
            if cost_key in cost_dict:
                _e = cost_dict[cost_key]
            else:
                _e = editdistance.eval(ref_spk_trans[idx][1], hyp_spk_trans[p][1])
                cost_dict[cost_key] = _e
            if _e > min_error > 0:
                flag = False
                break
            p_err += _e
            p_count += len(ref_spk_trans[idx][1])

        if flag:
            if p_err < min_error or min_error == 0:
                min_error = p_err

            errors.append(p_err)
            counts.append(p_count)
            permutes.append(perm)

    sd_cer = [(err, cnt, err/cnt, permute)
              for err, cnt, permute in zip(errors, counts, permutes)]
    best_rst = min(sd_cer, key=lambda x: x[2])

    return best_rst[0], best_rst[1], ref_spk_num, hyp_spk_num


def main():
    ref=sys.argv[1]
    hyp=sys.argv[2]
    result_path="/".join(hyp.split("/")[:-1]) + "/text_cpcer"
    ref_list = load_transcripts(ref)
    hyp_list = load_transcripts(hyp)
    result_file = open(result_path,'w')
    record_2_spk = [0, 0]
    record_3_spk = [0, 0]
    record_4_spk = [0, 0]
    error, count = 0, 0
    for (ref_id, ref_trans), (hyp_id, hyp_trans) in zip(ref_list, hyp_list):
        assert ref_id == hyp_id
        mid = ref_id
        dist, length, ref_spk_num, hyp_spk_num = calc_cer(ref_trans, hyp_trans)
        error, count = error + dist, count + length
        result_file.write("{} {:.2f} {} {}\n".format(mid, dist / length * 100.0, ref_spk_num, hyp_spk_num))
        ref_spk = len(ref_trans.split("$"))
        hyp_spk = len(hyp_trans.split("$"))
        if ref_spk == 2:
            record_2_spk[0] += dist
            record_2_spk[1] += length
        elif ref_spk == 3:
            record_3_spk[0] += dist
            record_3_spk[1] += length
        else:
            record_4_spk[0] += dist
            record_4_spk[1] += length
    print(record_2_spk[0]/record_2_spk[1]*100.0)
    print(record_3_spk[0]/record_3_spk[1]*100.0)
    print(record_4_spk[0]/record_4_spk[1]*100.0)
    result_file.write("CP-CER: {:.2f}\n".format(error / count * 100.0))
    result_file.close()


if __name__ == '__main__':
    main()
