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


@pytest.mark.parametrize("max_num_spks", [1, 3, 5, 8, 10])
@pytest.mark.parametrize("as_tensor", [False, True])
def test_large_auto_speaker_count_respects_configured_limit(monkeypatch, max_num_spks, as_tensor):
    backend = ClusterBackend()
    backend.spectral_cluster.max_num_spks = max_num_spks
    raw_labels = np.arange(2048) % 8
    embeddings = np.eye(8, dtype=np.float32)[raw_labels]
    if as_tensor:
        embeddings = torch.from_numpy(embeddings)
    monkeypatch.setattr(backend, "umap_hdbscan_cluster", lambda X: raw_labels.copy())

    labels = backend(embeddings)

    assert np.unique(labels).size == min(8, max_num_spks)


@pytest.mark.parametrize("max_num_spks", [1, 5, 16])
def test_speaker_cap_constructor_configures_large_auto_clustering(monkeypatch, max_num_spks):
    backend = ClusterBackend(max_num_spks=max_num_spks)
    raw_labels = np.arange(2048) % 20
    embeddings = np.eye(20, dtype=np.float32)[raw_labels]
    monkeypatch.setattr(backend, "umap_hdbscan_cluster", lambda X: raw_labels.copy())

    labels = backend(embeddings)

    assert backend.spectral_cluster.max_num_spks == max_num_spks
    assert np.unique(labels).size == max_num_spks


def test_default_speaker_cap_applies_to_large_auto_clustering(monkeypatch):
    backend = ClusterBackend()
    raw_labels = np.arange(2048) % 20
    embeddings = np.eye(20, dtype=np.float32)[raw_labels]
    monkeypatch.setattr(backend, "umap_hdbscan_cluster", lambda X: raw_labels.copy())

    assert np.unique(backend(embeddings)).size == 15


@pytest.mark.parametrize("max_num_spks", [0, -1, 1.5, True])
def test_invalid_speaker_cap_is_rejected(max_num_spks):
    with pytest.raises(ValueError, match="max_num_spks must be a positive integer"):
        ClusterBackend(max_num_spks=max_num_spks)


def test_speaker_cap_is_configurable_through_auto_model(monkeypatch):
    from funasr.auto.auto_model import AutoModel

    monkeypatch.setattr(
        AutoModel, "build_model", staticmethod(lambda **kwargs: (torch.nn.Identity(), kwargs))
    )
    spk_kwargs = {"cb_kwargs": {"max_num_spks": 3}}
    model = AutoModel(
        model="stub-asr",
        spk_model="stub-speaker",
        spk_kwargs=spk_kwargs,
        device="cpu",
        disable_update=True,
    )

    assert model.cb_model.spectral_cluster.max_num_spks == 3
    assert spk_kwargs == {"cb_kwargs": {"max_num_spks": 3}}


@pytest.mark.parametrize("num_embeddings", [64, 2048])
def test_known_speaker_count_overrides_automatic_cap(num_embeddings):
    backend = ClusterBackend(max_num_spks=1)
    expected = np.arange(num_embeddings) % 2
    embeddings = np.eye(2, dtype=np.float32)[expected]

    labels = backend(embeddings, oracle_num=2)

    np.testing.assert_array_equal(labels[:, None] == labels, expected[:, None] == expected)


@pytest.mark.parametrize("num_embeddings", [2047, 2048])
def test_automatic_clustering_dispatch_boundary(monkeypatch, num_embeddings):
    backend = ClusterBackend(max_num_spks=2)
    expected = np.arange(num_embeddings) % 2
    embeddings = np.eye(2, dtype=np.float32)[expected]
    calls = []

    def spectral(self, X, k):
        calls.append("spectral")
        assert k is None
        return expected.copy()

    def umap(X):
        calls.append("umap")
        return expected.copy()

    monkeypatch.setattr(type(backend.spectral_cluster), "__call__", spectral)
    monkeypatch.setattr(backend, "umap_hdbscan_cluster", umap)

    np.testing.assert_array_equal(backend(embeddings), expected)
    assert calls == ["spectral" if num_embeddings < 2048 else "umap"]


@pytest.mark.parametrize("max_num_spks", [2, 8])
def test_large_auto_speaker_cap_preserves_noise(monkeypatch, max_num_spks):
    backend = ClusterBackend(max_num_spks=max_num_spks)
    raw_labels = np.arange(2048) % 9 - 1
    embeddings = np.eye(8, dtype=np.float32)[raw_labels.clip(min=0)]
    monkeypatch.setattr(backend, "umap_hdbscan_cluster", lambda X: raw_labels.copy())

    labels = backend(embeddings)

    np.testing.assert_array_equal(labels == -1, raw_labels == -1)
    assert np.unique(labels[labels >= 0]).size == max_num_spks


def test_speaker_cap_merges_closest_centroids():
    embeddings = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.01, 1]], dtype=np.float32)
    original_labels = np.arange(4)

    labels = ClusterBackend().merge_to_max_spks(original_labels, embeddings, 3)

    assert np.unique(labels).size == 3
    assert labels[2] == labels[3]
    assert len(set(labels[:3])) == 3
    np.testing.assert_array_equal(original_labels, np.arange(4))


@pytest.mark.parametrize(
    "embeddings",
    [np.eye(3), np.array([[1, 0], [-1, 0], [0, 1]]), np.zeros((3, 2))],
)
def test_speaker_cap_merges_nonpositive_similarities_and_zero_centroids(embeddings):
    labels = ClusterBackend().merge_to_max_spks(np.arange(3), embeddings, 1)

    np.testing.assert_array_equal(labels, np.zeros(3, dtype=int))


@pytest.mark.parametrize(
    "labels", [np.array([], dtype=int), np.array([-1, -1]), np.array([-1, 0, 1])]
)
def test_speaker_cap_preserves_labels_already_within_limit(labels):
    embeddings = np.ones((len(labels), 2), dtype=np.float32)

    actual = ClusterBackend().merge_to_max_spks(labels, embeddings, 2)

    np.testing.assert_array_equal(actual, labels)
