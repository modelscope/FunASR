import sys


if __name__ == "__main__":
    path=sys.argv[1]
    text_scp_file = open(path + '/text', 'r')
    text_scp = text_scp_file.readlines()
    text_scp_file.close()
    text_id_scp_file = open(path + '/text_id', 'r')
    text_id_scp = text_id_scp_file.readlines()
    text_id_scp_file.close()
    text_spk_merge_file = open(path + '/text_spk_merge', 'w')
    assert len(text_scp) == len(text_id_scp)

    meeting_map = {} # {meeting_id: [(start_time, text, text_id), (start_time, text, text_id), ...]}
    for i in range(len(text_scp)):
        text_line = text_scp[i].strip().split(' ')
        text_id_line = text_id_scp[i].strip().split(' ')
        assert text_line[0] == text_id_line[0]
        if len(text_line) > 1:
            uttid = text_line[0]
            text = text_line[1]
            text_id = text_id_line[1]
            meeting_id = uttid.split('-')[0]
            start_time = int(uttid.split('-')[-2])
            if meeting_id not in meeting_map:
                meeting_map[meeting_id] = [(start_time,text,text_id)]
            else:
                meeting_map[meeting_id].append((start_time,text,text_id))
            
    for meeting_id in sorted(meeting_map.keys()):
        cur_meeting_list = sorted(meeting_map[meeting_id], key=lambda x: x[0])
        text_spk_merge_map = {} #{1: text1, 2: text2, ...}
        for cur_utt in cur_meeting_list:
            cur_text = cur_utt[1]
            cur_text_id = cur_utt[2]
            assert len(cur_text)==len(cur_text_id)
            if len(cur_text) != 0:
                cur_text_split = cur_text.split('$')
                cur_text_id_split = cur_text_id.split('$')
                assert len(cur_text_split) == len(cur_text_id_split)
                for i in range(len(cur_text_split)):
                    if len(cur_text_split[i]) != 0:
                        spk_id = int(cur_text_id_split[i][0])
                        if spk_id not in text_spk_merge_map.keys():
                            text_spk_merge_map[spk_id] = cur_text_split[i]
                        else:
                            text_spk_merge_map[spk_id] += cur_text_split[i]
        text_spk_merge_list = []
        for spk_id in sorted(text_spk_merge_map.keys()):
            text_spk_merge_list.append(text_spk_merge_map[spk_id])
        text_spk_merge_file.write(meeting_id + ' ' + '$'.join(text_spk_merge_list) + '\n')
        text_spk_merge_file.flush()
    
    text_spk_merge_file.close()