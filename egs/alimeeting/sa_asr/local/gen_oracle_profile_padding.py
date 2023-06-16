import random
import numpy as np
import os
import sys


if __name__=="__main__":
    path = sys.argv[1] 
    wav_scp_file = open(path+"/wav.scp", 'r')
    wav_scp = wav_scp_file.readlines()
    wav_scp_file.close()
    spk2id_file = open(path+"/spk2id", 'r')
    spk2id = spk2id_file.readlines()
    spk2id_file.close()
    embedding_scp_file = open(path + "/oracle_embedding.scp", 'r')
    embedding_scp = embedding_scp_file.readlines()
    embedding_scp_file.close()

    embedding_map = {}
    for line in embedding_scp:
        spk = line.strip().split(' ')[0]
        if spk not in embedding_map.keys():
            emb = np.load(line.strip().split(' ')[1])
            embedding_map[spk] = emb
    
    meeting_map_tmp = {}
    global_spk_list = []
    for line in spk2id:
        line_list = line.strip().split(' ')
        meeting = line_list[0].split('-')[0]
        spk_id = line_list[0].split('-')[-1].split('_')[-1]
        spk = meeting + '_' + spk_id
        global_spk_list.append(spk)
        if meeting in meeting_map_tmp.keys():
            meeting_map_tmp[meeting].append(spk)
        else:
            meeting_map_tmp[meeting] = [spk]
    
    for meeting in meeting_map_tmp.keys():
        num = len(meeting_map_tmp[meeting])
        if num < 4:
            global_spk_list_tmp = global_spk_list[: ]
            for spk in meeting_map_tmp[meeting]:
                global_spk_list_tmp.remove(spk)
            padding_spk = random.sample(global_spk_list_tmp, 4 - num)
            meeting_map_tmp[meeting] = meeting_map_tmp[meeting] + padding_spk
    
    meeting_map = {}
    os.system('mkdir -p ' + path + '/oracle_profile_padding')
    for meeting in meeting_map_tmp.keys():
        emb_list = []
        for i in range(len(meeting_map_tmp[meeting])):
            spk = meeting_map_tmp[meeting][i]
            emb_list.append(embedding_map[spk])
        profile = np.vstack(emb_list)
        np.save(path + '/oracle_profile_padding/' + meeting + '.npy',profile)
        meeting_map[meeting] = path + '/oracle_profile_padding/' + meeting + '.npy'
    
    profile_scp = open(path + '/oracle_profile_padding.scp', 'w')
    profile_map_scp = open(path + '/oracle_profile_padding_spk_list', 'w')

    for line in wav_scp:
        uttid = line.strip().split(' ')[0]
        meeting = uttid.split('-')[0]
        profile_scp.write(uttid+' ' + meeting_map[meeting] + '\n')
        profile_map_scp.write(uttid+' ' + '$'.join(meeting_map_tmp[meeting]) + '\n')
    profile_scp.close()
    profile_map_scp.close()