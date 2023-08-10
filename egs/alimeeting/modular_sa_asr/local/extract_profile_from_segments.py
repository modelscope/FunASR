import codecs
import sys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import os
import soundfile


data_path = sys.argv[1]
segment_file_path = data_path + "/segments_nooverlap"
utt2spk_file_path = data_path + "/utt2spk_nooverlap"
wav_scp_path = data_path + "/wav.scp"
cluster_emb_dir = data_path + '/cluster_embedding/'
os.system("mkdir -p " + cluster_emb_dir)
cluster_profile_dir = data_path + '/cluster_profile_zeropadding16/'
os.system('mkdir -p ' + cluster_profile_dir)

utt2spk = {}
spk2seg = {}
with codecs.open(utt2spk_file_path, "r", "utf-8") as f1:
    with codecs.open(segment_file_path, "r", "utf-8") as f2:
        for line in f1.readlines():
            uttid, spkid = line.strip().split(" ")
            utt2spk[uttid] = spkid
        for line in f2.readlines():
            uttid, sessionid, stime, etime = line.strip().split(" ")
            spkid = utt2spk[uttid]
            if spkid not in spk2seg.keys():
                spk2seg[spkid] = [(int(float(stime) * 16000), int(float(etime) * 16000) - int(float(stime) * 16000))]
            else:
                spk2seg[spkid].append((int(float(stime) * 16000), int(float(etime) * 16000) - int(float(stime) * 16000)))
          
inference_sv_pipline = pipeline(
    task=Tasks.speaker_verification,
    model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch',
    device='gpu'
)

wav_dict = {}

with codecs.open(wav_scp_path, "r", "utf-8") as fi:
    with codecs.open(data_path + "/cluster_embedding.scp", "w", "utf-8") as fo:
        for line in fi.readlines():
            sessionid, wav_path = line.strip().split()
            wav_dict[sessionid] = wav_path
        for spkid, segs in spk2seg.items():
            sessionid = spkid.split("-")[0]
            wav_path = wav_dict[sessionid]
            wav = soundfile.read(wav_path)[0]
            if wav.ndim == 2:
                wav = wav[:, 0]
            all_seg_embedding_list=[]
            for seg in segs:
                if seg[0] < wav.shape[0] - 0.5 * 16000:
                    if seg[1] > wav.shape[0]:
                        cur_seg_embedding = inference_sv_pipline(audio_in=wav[seg[0]: ])["spk_embedding"]
                    else:
                        cur_seg_embedding = inference_sv_pipline(audio_in=wav[seg[0]: seg[0] + seg[1]])["spk_embedding"]
                    all_seg_embedding_list.append(cur_seg_embedding)
            all_seg_embedding = np.vstack(all_seg_embedding_list)
            spk_embedding = np.mean(all_seg_embedding, axis=0)
            np.save(cluster_emb_dir + spkid + '.npy', spk_embedding)    
            fo.write(spkid + ' ' + cluster_emb_dir + spkid + '.npy' + '\n')

session2embs = {}

with codecs.open(data_path + "/cluster_embedding.scp", "r", "utf-8") as fi:
    with codecs.open(data_path + "/cluster_profile_zeropadding16.scp", "w", "utf-8") as fo:
        for line in fi.readlines():
            spkid, emb_path = line.strip().split(" ")
            sessionid = spkid.split("-")[0]
            if sessionid not in session2embs.keys():
                session2embs[sessionid] = [emb_path]
            else:
                session2embs[sessionid].append(emb_path)
        for sessionid, embs in session2embs.items():
            emb_list = [np.load(x) for x in embs]
            tmp = []
            for i in range(len(emb_list) - 1):
                flag = True
                for j in range(i + 1, len(emb_list)):
                    cos_sim = emb_list[i].dot(emb_list[j]) / (np.linalg.norm(emb_list[i]) * np.linalg.norm(emb_list[j]))
                    if cos_sim > 0.99:
                        flag = False
                if flag:
                    tmp.append(emb_list[i][np.newaxis, :])
            tmp.append(emb_list[-1][np.newaxis, :])
            emb_list = tmp
            # tmp = []
            # for i in range(len(emb_list)):
            #     for emb in tmp:
            #         cos_sim = emb_list[i].dot(emb_list[j]) / (np.linalg.norm(emb_list[i]) * np.linalg.norm(emb_list[j]))
            #         if cos_sim > 0.99:
            #             flag = False
            #     if flag:
            #         tmp.append(emb_list[i][np.newaxis, :])
            # emb_list = tmp
            for i in range(16 - len(emb_list)):
                emb_list.append(np.zeros((1, 256)))
            emb = np.concatenate(emb_list, axis=0)
            save_path = cluster_profile_dir + sessionid + ".npy"
            np.save(save_path, emb)
            fo.write("%s %s\n" % (sessionid, save_path))