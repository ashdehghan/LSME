"""Test script for SignatureEncoder module."""

import os
import tempfile

import numpy as np
import networkx as nx
import torch

from lsme import LSME, SignatureEncoder


def test_encoder_basic():
    """Test 1: Basic encoding workflow."""
    print("Test 1: Basic encoding workflow")
    print("-" * 50)

    # Generate signature matrices
    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    # Train encoder
    encoder = SignatureEncoder(
        embedding_dim=32,
        max_matrix_size=32,
        num_epochs=20,
        verbose=False,
        random_state=42
    )

    embeddings = encoder.fit_transform(
        result['signature_matrices'],
        result['layer_info']
    )

    # Verify output structure
    assert isinstance(embeddings, dict), "Embeddings should be dict"
    assert len(embeddings) == G.number_of_nodes(), "Should have embedding per node"

    for node_id, emb in embeddings.items():
        assert isinstance(emb, np.ndarray), "Embedding should be numpy array"
        assert emb.shape == (32,), f"Embedding shape should be (32,), got {emb.shape}"

    print(f"Generated {len(embeddings)} embeddings of shape (32,)")
    print("Test 1 passed!\n")


def test_encode_decode():
    """Test 2: Encode-decode roundtrip."""
    print("Test 2: Encode-decode roundtrip")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    encoder = SignatureEncoder(
        embedding_dim=64,
        num_epochs=50,
        verbose=False,
        random_state=42
    )
    encoder.fit(result['signature_matrices'], result['layer_info'])

    # Encode then decode
    embeddings = encoder.encode(result['signature_matrices'], result['layer_info'])
    reconstructed = encoder.decode(embeddings)

    # Verify reconstruction shapes
    for node_id in result['signature_matrices']:
        recon = reconstructed[node_id]
        # Reconstructed should be padded size (square)
        assert recon.shape[0] == recon.shape[1], "Reconstruction should be square"
        # Values should be in [0, 1] (sigmoid output)
        assert recon.min() >= 0.0, "Reconstruction values should be >= 0"
        assert recon.max() <= 1.0, "Reconstruction values should be <= 1"

    print("Encode-decode roundtrip successful")
    print("Test 2 passed!\n")


def test_variable_sizes():
    """Test 3: Variable matrix sizes."""
    print("Test 3: Variable matrix sizes")
    print("-" * 50)

    # Create graph with varying neighborhood sizes
    # Star graph: center has many neighbors, leaves have few at 1-hop
    G = nx.star_graph(9)  # Creates star with center 0 and leaves 1-9
    # Add some extra structure
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])

    # Use max_hops=1 to create varying sizes (center sees 9, leaves see 1-3)
    lsme = LSME(max_hops=1, n_samples=30, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    # Check that we have varying sizes
    sizes = [info['total_nodes'] for info in result['layer_info'].values()]
    assert len(set(sizes)) > 1, "Test graph should have varying neighborhood sizes"

    encoder = SignatureEncoder(
        embedding_dim=16,
        max_matrix_size=32,
        num_epochs=20,
        verbose=False,
        random_state=42
    )
    embeddings = encoder.fit_transform(
        result['signature_matrices'],
        result['layer_info']
    )

    # All embeddings should have same dimension regardless of input size
    for emb in embeddings.values():
        assert emb.shape == (16,), "All embeddings should have same dimension"

    print(f"Matrix sizes in test: {sorted(set(sizes))}")
    print("All embeddings have uniform dimension (16,)")
    print("Test 3 passed!\n")


def test_reproducibility():
    """Test 4: Reproducibility with random_state."""
    print("Test 4: Reproducibility")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    # Train twice with same random state
    encoder1 = SignatureEncoder(
        embedding_dim=32,
        num_epochs=30,
        verbose=False,
        random_state=123
    )
    emb1 = encoder1.fit_transform(
        result['signature_matrices'],
        result['layer_info']
    )

    encoder2 = SignatureEncoder(
        embedding_dim=32,
        num_epochs=30,
        verbose=False,
        random_state=123
    )
    emb2 = encoder2.fit_transform(
        result['signature_matrices'],
        result['layer_info']
    )

    for node in emb1:
        assert np.allclose(emb1[node], emb2[node], atol=1e-5), \
            f"Embeddings should match for node {node}"

    print("Same random_state produces identical embeddings")
    print("Test 4 passed!\n")


def test_save_load():
    """Test 5: Model save/load."""
    print("Test 5: Model save/load")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    encoder = SignatureEncoder(
        embedding_dim=32,
        num_epochs=20,
        verbose=False,
        random_state=42
    )
    encoder.fit(result['signature_matrices'], result['layer_info'])
    emb_before = encoder.encode(result['signature_matrices'], result['layer_info'])

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "encoder.pt")
        encoder.save(path)

        loaded_encoder = SignatureEncoder.load(path)
        emb_after = loaded_encoder.encode(
            result['signature_matrices'],
            result['layer_info']
        )

    for node in emb_before:
        assert np.allclose(emb_before[node], emb_after[node]), \
            f"Embeddings should match after load for node {node}"

    print("Model save/load produces identical embeddings")
    print("Test 5 passed!\n")


def test_reconstruction_error():
    """Test 6: Reconstruction error computation."""
    print("Test 6: Reconstruction error")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    encoder = SignatureEncoder(
        embedding_dim=64,
        num_epochs=50,
        verbose=False,
        random_state=42
    )
    encoder.fit(result['signature_matrices'], result['layer_info'])

    errors = encoder.reconstruction_error(
        result['signature_matrices'],
        result['layer_info']
    )

    assert len(errors) == G.number_of_nodes(), "Should have error per node"
    for node_id, error in errors.items():
        assert isinstance(error, float), "Error should be float"
        assert error >= 0, "MSE should be non-negative"

    avg_error = np.mean(list(errors.values()))
    print(f"Average reconstruction error: {avg_error:.6f}")
    print("Test 6 passed!\n")


def test_gpu_support():
    """Test 7: GPU support (if available)."""
    print("Test 7: GPU support")
    print("-" * 50)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        print("Test 7 skipped!\n")
        return

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    encoder = SignatureEncoder(
        embedding_dim=32,
        num_epochs=10,
        device='cuda',
        verbose=True
    )
    embeddings = encoder.fit_transform(
        result['signature_matrices'],
        result['layer_info']
    )

    assert len(embeddings) == G.number_of_nodes()
    print("Test 7 passed!\n")


def test_small_graph():
    """Test 8: Small graph edge case."""
    print("Test 8: Small graph")
    print("-" * 50)

    # Very small graph
    G = nx.path_graph(3)  # 0 -- 1 -- 2

    lsme = LSME(max_hops=2, n_samples=30, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    encoder = SignatureEncoder(
        embedding_dim=8,
        max_matrix_size=16,
        num_epochs=20,
        verbose=False,
        random_state=42
    )
    embeddings = encoder.fit_transform(
        result['signature_matrices'],
        result['layer_info']
    )

    assert len(embeddings) == 3, "Should have 3 embeddings"
    for emb in embeddings.values():
        assert emb.shape == (8,), "Embedding dimension should be 8"

    print("Small graph handled correctly")
    print("Test 8 passed!\n")


if __name__ == "__main__":
    test_encoder_basic()
    test_encode_decode()
    test_variable_sizes()
    test_reproducibility()
    test_save_load()
    test_reconstruction_error()
    test_gpu_support()
    test_small_graph()
    print("=" * 50)
    print("All tests passed successfully!")
