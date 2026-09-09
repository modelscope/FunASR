"""Model-free compatibility checks for the production speaker clustering paths."""

import numpy as np
import pytest

from funasr.models.campplus.cluster_backend import ClusterBackend, UmapHdbscan


@pytest.fixture(autouse=True)
def preserve_numpy_random_state():
    # The production sklearn/UMAP calls use NumPy's global random state.
    state = np.random.get_state()
    np.random.seed(17)
    try:
        yield
    finally:
        np.random.set_state(state)


def _speaker_embeddings(num_speakers, dtype, samples_per_speaker=24):
    rng = np.random.default_rng(20260909)
    expected = np.repeat(np.arange(num_speakers), samples_per_speaker)
    centers = np.eye(num_speakers, 8)
    embeddings = centers[expected] + rng.normal(0.0, 0.015, (len(expected), 8))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    order = rng.permutation(len(expected))
    return embeddings[order].astype(dtype), expected[order]


def _assert_same_partition(labels, expected):
    labels = np.asarray(labels)
    assert labels.shape == expected.shape
    assert np.issubdtype(labels.dtype, np.integer)
    assert np.all(labels >= 0), "Separated speakers must not be classified as noise"
    np.testing.assert_array_equal(
        labels[:, None] == labels[None, :],
        expected[:, None] == expected[None, :],
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("num_speakers", [2, 3])
def test_spectral_partition_survives_dtype_and_input_order(dtype, num_speakers):
    embeddings, expected = _speaker_embeddings(num_speakers, dtype)
    original = embeddings.copy()
    # This range selects the real spectral branch, not the <20 zero-label path.
    assert 20 <= len(embeddings) < 2048
    backend = ClusterBackend()
    labels = backend(embeddings, oracle_num=num_speakers)
    _assert_same_partition(labels, expected)

    permutation = np.random.default_rng(91).permutation(len(embeddings))
    permuted_labels = backend(embeddings[permutation], oracle_num=num_speakers)
    _assert_same_partition(permuted_labels, expected[permutation])
    np.testing.assert_array_equal(embeddings, original)


def test_spectral_oracle_can_override_two_natural_speakers():
    embeddings, expected = _speaker_embeddings(2, np.float32)
    assert np.unique(expected).size == 2
    labels = ClusterBackend()(embeddings, oracle_num=1)
    _assert_same_partition(labels, np.zeros_like(expected))


def test_umap_hdbscan_preserves_separated_speakers():
    import numba

    embeddings, expected = _speaker_embeddings(2, np.float32, samples_per_speaker=32)
    original = embeddings.copy()
    # Exercise the real wrapper on a small fixture, not the >=2048 dispatcher.
    cluster = UmapHdbscan(
        n_neighbors=10, n_components=2, min_samples=3, min_cluster_size=12
    )
    previous_threads = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        labels = cluster(embeddings)
    finally:
        numba.set_num_threads(previous_threads)
    _assert_same_partition(labels, expected)
    np.testing.assert_array_equal(embeddings, original)
