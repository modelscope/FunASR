#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)

import scipy
import torch
import sklearn
import numpy as np

from sklearn.cluster._kmeans import k_means
from sklearn.cluster import HDBSCAN


class SpectralCluster:
    r"""A spectral clustering mehtod using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    """

    def __init__(self, min_num_spks=1, max_num_spks=15, pval=0.022):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.pval = pval

    def __call__(self, X, oracle_num=None):
        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with pval
        prunned_sim_mat = self.p_pruning(sim_mat)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)

        # Perform clustering
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def get_sim_mat(self, X):
        # Cosine similarities
        M = sklearn.metrics.pairwise.cosine_similarity(X, X)
        return M

    def p_pruning(self, A):
        if A.shape[0] * self.pval < 6:
            pval = 6.0 / A.shape[0]
        else:
            pval = self.pval

        n_elems = int((1 - pval) * A.shape[0])

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=None):
        lambdas, eig_vecs = scipy.linalg.eigh(L)

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1 : self.max_num_spks + 1]
            )
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        _, labels, _ = k_means(emb, k)
        return labels

    def getEigenGaps(self, eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


class UmapHdbscan:
    r"""
    Reference:
    - Siqi Zheng, Hongbin Suo. Reformulating Speaker Diarization as Community Detection With
      Emphasis On Topological Structure. ICASSP2022
    """

    def __init__(
        self, n_neighbors=20, n_components=60, min_samples=10, min_cluster_size=10, metric="cosine"
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.metric = metric

    def __call__(self, X):
        import umap.umap_ as umap

        umap_X = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=0.0,
            n_components=min(self.n_components, X.shape[0] - 2),
            metric=self.metric,
        ).fit_transform(X)
        labels = HDBSCAN(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            allow_single_cluster=True,
        ).fit_predict(umap_X)
        return labels


class ClusterBackend(torch.nn.Module):
    r"""Perfom clustering for input embeddings and output the labels.
    Args:
        model_dir: A model dir.
        model_config: The model config.
    """

    def __init__(self, merge_thr=0.78):
        super().__init__()
        self.model_config = {"merge_thr": merge_thr}
        # self.other_config = kwargs

        self.spectral_cluster = SpectralCluster()
        self.umap_hdbscan_cluster = UmapHdbscan()

    def forward(self, X, **params):
        # clustering and return the labels
        k = params["oracle_num"] if "oracle_num" in params else None
        assert len(X.shape) == 2, "modelscope error: the shape of input should be [N, C]"
        if X.shape[0] < 20:
            return np.zeros(X.shape[0], dtype="int")
        if X.shape[0] < 2048 or k is not None:
            # unexpected corner case
            labels = self.spectral_cluster(X, k)
        else:
            labels = self.umap_hdbscan_cluster(X)

        if k is None and "merge_thr" in self.model_config:
            labels = self.merge_by_cos(labels, X, self.model_config["merge_thr"])

        return labels

    def merge_by_cos(self, labels, embs, cos_thr):
        # merge the similar speakers by cosine similarity
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            spk_num = labels.max() + 1
            if spk_num == 1:
                break
            spk_center = []
            for i in range(spk_num):
                spk_emb = embs[labels == i].mean(0)
                spk_center.append(spk_emb)
            assert len(spk_center) > 0
            spk_center = np.stack(spk_center, axis=0)
            norm_spk_center = spk_center / np.linalg.norm(spk_center, axis=1, keepdims=True)
            affinity = np.matmul(norm_spk_center, norm_spk_center.T)
            affinity = np.triu(affinity, 1)
            spks = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[spks] < cos_thr:
                break
            for i in range(len(labels)):
                if labels[i] == spks[1]:
                    labels[i] = spks[0]
                elif labels[i] > spks[1]:
                    labels[i] -= 1
        return labels
