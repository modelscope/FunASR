from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import sys
import os
import soundfile
from itertools import permutations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import cluster


def custom_spectral_clustering(affinity, min_n_clusters=2, max_n_clusters=4, refine=True,
    threshold=0.995, laplacian_type="graph_cut"):
    if refine:
        # Symmetrization
        affinity = np.maximum(affinity, np.transpose(affinity))
        # Diffusion
        affinity = np.matmul(affinity, np.transpose(affinity))
        # Row-wise max normalization
        row_max = affinity.max(axis=1, keepdims=True)
        affinity = affinity / row_max

    # a) Construct S and set diagonal elements to 0
    affinity = affinity - np.diag(np.diag(affinity))
    # b) Compute Laplacian matrix L and perform normalization:
    degree = np.diag(np.sum(affinity, axis=1))
    laplacian = degree - affinity
    if laplacian_type == "random_walk":
        degree_norm = np.diag(1 / (np.diag(degree) + 1e-10))
        laplacian_norm = degree_norm.dot(laplacian)
    else:
        degree_half = np.diag(degree) ** 0.5 + 1e-15
        laplacian_norm = laplacian / degree_half[:, np.newaxis] / degree_half

    # c) Compute eigenvalues and eigenvectors of L_norm
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_norm)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    index_array = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[index_array]
    eigenvectors = eigenvectors[:, index_array]

    # d) Compute the number of clusters k
    k = min_n_clusters
    for k in range(min_n_clusters, max_n_clusters + 1):
        if eigenvalues[k] > threshold:
            break
    k = max(k, min_n_clusters)
    spectral_embeddings = eigenvectors[:, :k]
    # print(mid, k, eigenvalues[:10])

    spectral_embeddings = spectral_embeddings / np.linalg.norm(spectral_embeddings, axis=1, ord=2, keepdims=True)
    solver = cluster.KMeans(n_clusters=k, max_iter=1000, random_state=42)
    solver.fit(spectral_embeddings)
    return solver.labels_


if __name__ == "__main__":
    path = sys.argv[1] # dump2/raw/Eval_Ali_far
    raw_path = sys.argv[2] # data/local/Eval_Ali_far
    threshold = float(sys.argv[3]) # 0.996
    sv_threshold = float(sys.argv[4]) # 0.815
    wav_scp_file = open(path+'/wav.scp', 'r')
    wav_scp = wav_scp_file.readlines()
    wav_scp_file.close()
    raw_meeting_scp_file = open(raw_path + '/wav.scp', 'r')
    raw_meeting_scp = raw_meeting_scp_file.readlines()
    raw_meeting_scp_file.close()
    segments_scp_file = open(raw_path + '/segments', 'r')
    segments_scp = segments_scp_file.readlines()
    segments_scp_file.close()

    segments_map = {}
    for line in segments_scp:
        line_list = line.strip().split(' ')
        meeting = line_list[1]
        seg = (float(line_list[-2]), float(line_list[-1]))
        if meeting not in segments_map.keys():
            segments_map[meeting] = [seg]
        else:
            segments_map[meeting].append(seg)
    
    inference_sv_pipline = pipeline(
        task=Tasks.speaker_verification,
        model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch'
    )

    chunk_len = int(1.5*16000) # 1.5 seconds
    hop_len = int(0.75*16000) # 0.75 seconds

    os.system("mkdir -p " + path + "/cluster_profile_infer")
    cluster_spk_num_file = open(path + '/cluster_spk_num', 'w')
    meeting_map = {}
    for line in raw_meeting_scp:
        meeting = line.strip().split(' ')[0]
        wav_path = line.strip().split(' ')[1]
        wav = soundfile.read(wav_path)[0]
        # take the first channel
        if wav.ndim == 2:
            wav=wav[:, 0]
        # gen_seg_embedding
        segments_list = segments_map[meeting]

        # import ipdb;ipdb.set_trace()
        all_seg_embedding_list = []
        for seg in segments_list:
            wav_seg = wav[int(seg[0] * 16000): int(seg[1] * 16000)]
            wav_seg_len = wav_seg.shape[0]
            i = 0
            while i < wav_seg_len:
                if i + chunk_len < wav_seg_len:
                    cur_wav_chunk = wav_seg[i: i+chunk_len]
                else:
                    cur_wav_chunk=wav_seg[i: ]
                # chunks under 0.2s are ignored
                if cur_wav_chunk.shape[0] >= 0.2 * 16000:
                    cur_chunk_embedding = inference_sv_pipline(audio_in=cur_wav_chunk)["spk_embedding"]
                    all_seg_embedding_list.append(cur_chunk_embedding)
                i += hop_len
        all_seg_embedding = np.vstack(all_seg_embedding_list)
        # all_seg_embedding (n, dim)

        # compute affinity
        affinity=cosine_similarity(all_seg_embedding)

        affinity = np.maximum(affinity - sv_threshold, 0.0001) / (affinity.max() - sv_threshold)

        # clustering
        labels = custom_spectral_clustering(
            affinity=affinity,
            min_n_clusters=2,
            max_n_clusters=4,
            refine=True,
            threshold=threshold,
            laplacian_type="graph_cut")
       

        cluster_dict={}
        for j in range(labels.shape[0]):
            if labels[j] not in cluster_dict.keys():
                cluster_dict[labels[j]] = np.atleast_2d(all_seg_embedding[j])
            else:
                cluster_dict[labels[j]] = np.concatenate((cluster_dict[labels[j]], np.atleast_2d(all_seg_embedding[j])))
        
        emb_list = []
        # get cluster center
        for k in cluster_dict.keys():
            cluster_dict[k] = np.mean(cluster_dict[k], axis=0)
            emb_list.append(cluster_dict[k])

        spk_num = len(emb_list)
        profile_for_infer = np.vstack(emb_list)
        # save profile for each meeting
        np.save(path + '/cluster_profile_infer/' + meeting + '.npy', profile_for_infer)
        meeting_map[meeting] = (path + '/cluster_profile_infer/' + meeting + '.npy', spk_num)
        cluster_spk_num_file.write(meeting + ' ' + str(spk_num) + '\n')
        cluster_spk_num_file.flush()
    
    cluster_spk_num_file.close()

    profile_scp = open(path + "/cluster_profile_infer.scp", 'w')
    for line in wav_scp:
        uttid = line.strip().split(' ')[0]
        meeting = uttid.split('-')[0]
        profile_scp.write(uttid + ' ' + meeting_map[meeting][0] + '\n')
        profile_scp.flush()
    profile_scp.close()
