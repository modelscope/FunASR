import numpy as np
import pytest
import torch

from funasr.models.campplus.cluster_backend import ClusterBackend


def test_large_known_speaker_count_uses_fixed_k_clustering(monkeypatch):
    backend = ClusterBackend()
    embeddings = torch.ones((2048, 2))
    expected = np.arange(embeddings.shape[0]) % 2

    def fail_spectral(*args, **kwargs):
        pytest.fail("large inputs must not use dense spectral clustering")

    def fail_umap(*args, **kwargs):
        pytest.fail("a known speaker count should use fixed-K clustering")

    def fixed_k_cluster(actual_embeddings, num_clusters):
        assert actual_embeddings is embeddings
        assert num_clusters == 2
        return expected

    monkeypatch.setattr(backend, "spectral_cluster", fail_spectral)
    monkeypatch.setattr(backend, "umap_hdbscan_cluster", fail_umap)
    monkeypatch.setattr(backend, "kmeans_cluster", fixed_k_cluster, raising=False)

    labels = backend(embeddings, oracle_num=2)

    np.testing.assert_array_equal(labels, expected)


def test_large_fixed_k_clustering_separates_cosine_clusters():
    backend = ClusterBackend()
    embeddings = torch.zeros((2048, 2))
    embeddings[:1024, 0] = 1
    embeddings[1024:, 1] = 1

    labels = backend(embeddings, oracle_num=2)

    assert np.unique(labels).size == 2
    assert np.unique(labels[:1024]).size == 1
    assert np.unique(labels[1024:]).size == 1
    assert labels[0] != labels[-1]
