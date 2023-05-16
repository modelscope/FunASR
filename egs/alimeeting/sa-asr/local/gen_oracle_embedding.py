from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import sys
import os
import soundfile


if __name__=="__main__":
    path = sys.argv[1] # dump2/raw/Eval_Ali_far
    raw_path = sys.argv[2] # data/local/Eval_Ali_far_correct_single_speaker
    raw_meeting_scp_file = open(raw_path + '/wav_raw.scp', 'r')
    raw_meeting_scp = raw_meeting_scp_file.readlines()
    raw_meeting_scp_file.close()
    segments_scp_file = open(raw_path + '/segments', 'r')
    segments_scp = segments_scp_file.readlines()
    segments_scp_file.close()

    oracle_emb_dir = path + '/oracle_embedding/'
    os.system("mkdir -p " + oracle_emb_dir)
    oracle_emb_scp_file = open(path+'/oracle_embedding.scp', 'w')

    raw_wav_map = {}
    for line in raw_meeting_scp:
        meeting = line.strip().split('\t')[0]
        wav_path = line.strip().split('\t')[1]
        raw_wav_map[meeting] = wav_path
    
    spk_map = {}
    for line in segments_scp:
        line_list = line.strip().split(' ')
        meeting = line_list[1]
        spk_id = line_list[0].split('_')[3]
        spk = meeting + '_' + spk_id
        time_start = float(line_list[-2])
        time_end = float(line_list[-1])
        if time_end - time_start > 0.5:
            if spk not in spk_map.keys():
                spk_map[spk] = [(int(time_start * 16000), int(time_end * 16000))]
            else:
                spk_map[spk].append((int(time_start * 16000), int(time_end * 16000)))
    
    inference_sv_pipline = pipeline(
        task=Tasks.speaker_verification,
        model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
    )

    for spk in spk_map.keys():
        meeting = spk.split('_SPK')[0]
        wav_path = raw_wav_map[meeting]
        wav = soundfile.read(wav_path)[0]
        # take the first channel
        if wav.ndim == 2:
            wav = wav[:, 0]
        all_seg_embedding_list=[]
        # import ipdb;ipdb.set_trace()
        for seg_time in spk_map[spk]:
            if seg_time[0] < wav.shape[0] - 0.5 * 16000:
                if seg_time[1] > wav.shape[0]:
                    cur_seg_embedding = inference_sv_pipline(audio_in=wav[seg_time[0]: ])["spk_embedding"]
                else:
                    cur_seg_embedding = inference_sv_pipline(audio_in=wav[seg_time[0]: seg_time[1]])["spk_embedding"]
                all_seg_embedding_list.append(cur_seg_embedding)
        all_seg_embedding = np.vstack(all_seg_embedding_list)
        spk_embedding = np.mean(all_seg_embedding, axis=0)
        np.save(oracle_emb_dir + spk + '.npy', spk_embedding)
        oracle_emb_scp_file.write(spk + ' ' + oracle_emb_dir + spk + '.npy' + '\n')
        oracle_emb_scp_file.flush()
    
    oracle_emb_scp_file.close()