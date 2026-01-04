"""Test script for encoder modules (CNNEncoder and DNNEncoder)."""

import os
import tempfile

import numpy as np
import networkx as nx
import torch

from lsme import LSME, SignatureEncoder, CNNEncoder, DNNEncoder


def test_cnn_encoder_basic():
    """Test 1: Basic CNN encoding workflow."""
    print("Test 1: CNN encoder basic workflow")
    print("-" * 50)

    # Generate signature matrices
    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    # Train encoder
    encoder = CNNEncoder(
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


def test_dnn_encoder_basic():
    """Test 2: Basic DNN encoding workflow."""
    print("Test 2: DNN encoder basic workflow")
    print("-" * 50)

    # Generate signature matrices
    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    # Train DNN encoder
    encoder = DNNEncoder(
        embedding_dim=32,
        max_matrix_size=32,
        hidden_dims=[256, 128, 64],
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

    print(f"Generated {len(embeddings)} DNN embeddings of shape (32,)")
    print("Test 2 passed!\n")


def test_encode_decode():
    """Test 3: Encode-decode roundtrip for both encoders."""
    print("Test 3: Encode-decode roundtrip")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    for encoder_cls, name in [(CNNEncoder, "CNN"), (DNNEncoder, "DNN")]:
        encoder = encoder_cls(
            embedding_dim=64,
            num_epochs=30,
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
            assert recon.shape[0] == recon.shape[1], f"{name}: Reconstruction should be square"
            assert recon.min() >= 0.0, f"{name}: Reconstruction values should be >= 0"
            assert recon.max() <= 1.0, f"{name}: Reconstruction values should be <= 1"

        print(f"  {name}: roundtrip OK")

    print("Test 3 passed!\n")


def test_variable_sizes():
    """Test 4: Variable matrix sizes."""
    print("Test 4: Variable matrix sizes")
    print("-" * 50)

    G = nx.star_graph(9)
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])

    lsme = LSME(max_hops=1, n_samples=30, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    sizes = [info['total_nodes'] for info in result['layer_info'].values()]
    assert len(set(sizes)) > 1, "Test graph should have varying neighborhood sizes"

    for encoder_cls, name in [(CNNEncoder, "CNN"), (DNNEncoder, "DNN")]:
        encoder = encoder_cls(
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

        for emb in embeddings.values():
            assert emb.shape == (16,), f"{name}: All embeddings should have dim 16"

        print(f"  {name}: handles variable sizes OK")

    print(f"Matrix sizes in test: {sorted(set(sizes))}")
    print("Test 4 passed!\n")


def test_reproducibility():
    """Test 5: Reproducibility with random_state."""
    print("Test 5: Reproducibility")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    for encoder_cls, name in [(CNNEncoder, "CNN"), (DNNEncoder, "DNN")]:
        encoder1 = encoder_cls(embedding_dim=32, num_epochs=20, verbose=False, random_state=123)
        emb1 = encoder1.fit_transform(result['signature_matrices'], result['layer_info'])

        encoder2 = encoder_cls(embedding_dim=32, num_epochs=20, verbose=False, random_state=123)
        emb2 = encoder2.fit_transform(result['signature_matrices'], result['layer_info'])

        for node in emb1:
            assert np.allclose(emb1[node], emb2[node], atol=1e-5), \
                f"{name}: Embeddings should match for node {node}"

        print(f"  {name}: reproducible")

    print("Test 5 passed!\n")


def test_save_load():
    """Test 6: Model save/load for both encoders."""
    print("Test 6: Model save/load")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    for encoder_cls, name in [(CNNEncoder, "CNN"), (DNNEncoder, "DNN")]:
        encoder = encoder_cls(embedding_dim=32, num_epochs=20, verbose=False, random_state=42)
        encoder.fit(result['signature_matrices'], result['layer_info'])
        emb_before = encoder.encode(result['signature_matrices'], result['layer_info'])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"{name.lower()}_encoder.pt")
            encoder.save(path)

            loaded_encoder = encoder_cls.load(path)
            emb_after = loaded_encoder.encode(result['signature_matrices'], result['layer_info'])

        for node in emb_before:
            assert np.allclose(emb_before[node], emb_after[node]), \
                f"{name}: Embeddings should match after load"

        print(f"  {name}: save/load OK")

    print("Test 6 passed!\n")


def test_reconstruction_error():
    """Test 7: Reconstruction error computation."""
    print("Test 7: Reconstruction error")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    for encoder_cls, name in [(CNNEncoder, "CNN"), (DNNEncoder, "DNN")]:
        encoder = encoder_cls(embedding_dim=64, num_epochs=50, verbose=False, random_state=42)
        encoder.fit(result['signature_matrices'], result['layer_info'])

        errors = encoder.reconstruction_error(result['signature_matrices'], result['layer_info'])

        assert len(errors) == G.number_of_nodes(), f"{name}: Should have error per node"
        for node_id, error in errors.items():
            assert isinstance(error, float), f"{name}: Error should be float"
            assert error >= 0, f"{name}: MSE should be non-negative"

        avg_error = np.mean(list(errors.values()))
        print(f"  {name}: avg error = {avg_error:.6f}")

    print("Test 7 passed!\n")


def test_gpu_support():
    """Test 8: GPU support (if available)."""
    print("Test 8: GPU support")
    print("-" * 50)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        print("Test 8 skipped!\n")
        return

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    for encoder_cls, name in [(CNNEncoder, "CNN"), (DNNEncoder, "DNN")]:
        encoder = encoder_cls(embedding_dim=32, num_epochs=10, device='cuda', verbose=True)
        embeddings = encoder.fit_transform(result['signature_matrices'], result['layer_info'])

        assert len(embeddings) == G.number_of_nodes()
        print(f"  {name}: GPU OK")

    print("Test 8 passed!\n")


def test_small_graph():
    """Test 9: Small graph edge case."""
    print("Test 9: Small graph")
    print("-" * 50)

    G = nx.path_graph(3)

    lsme = LSME(max_hops=2, n_samples=30, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    for encoder_cls, name in [(CNNEncoder, "CNN"), (DNNEncoder, "DNN")]:
        encoder = encoder_cls(
            embedding_dim=8,
            max_matrix_size=16,
            num_epochs=20,
            verbose=False,
            random_state=42
        )
        embeddings = encoder.fit_transform(result['signature_matrices'], result['layer_info'])

        assert len(embeddings) == 3, f"{name}: Should have 3 embeddings"
        for emb in embeddings.values():
            assert emb.shape == (8,), f"{name}: Embedding dimension should be 8"

        print(f"  {name}: small graph OK")

    print("Test 9 passed!\n")


def test_signature_encoder_alias():
    """Test 10: SignatureEncoder is alias for CNNEncoder."""
    print("Test 10: SignatureEncoder alias")
    print("-" * 50)

    assert SignatureEncoder is CNNEncoder, "SignatureEncoder should be CNNEncoder"

    G = nx.karate_club_graph()
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    encoder = SignatureEncoder(embedding_dim=16, num_epochs=10, verbose=False, random_state=42)
    embeddings = encoder.fit_transform(result['signature_matrices'], result['layer_info'])

    assert len(embeddings) == G.number_of_nodes()
    print("SignatureEncoder works as CNNEncoder alias")
    print("Test 10 passed!\n")


if __name__ == "__main__":
    test_cnn_encoder_basic()
    test_dnn_encoder_basic()
    test_encode_decode()
    test_variable_sizes()
    test_reproducibility()
    test_save_load()
    test_reconstruction_error()
    test_gpu_support()
    test_small_graph()
    test_signature_encoder_alias()
    print("=" * 50)
    print("All encoder tests passed successfully!")
