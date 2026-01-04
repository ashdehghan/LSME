"""Test script to verify LSME package functionality."""

import networkx as nx
import numpy as np
from lsme import LSME


def test_basic_usage():
    """Test 1: Basic usage with Karate Club graph (default stochastic method)."""
    print("Test 1: Basic usage (stochastic)")
    print("-" * 50)

    G = nx.karate_club_graph()
    print(f"Graph: Karate Club")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    embedding_dim = 32
    embedder = LSME(max_hops=2, n_samples=50, embedding_dim=embedding_dim,
                    encoder_epochs=20, verbose=False)
    result = embedder.fit_transform(G)

    # Verify return structure (new unified format)
    assert isinstance(result, dict), "Result should be a dict"
    assert "embeddings" in result, "Missing embeddings key"
    assert "method" in result, "Missing method key"
    assert "params" in result, "Missing params key"
    assert "metadata" in result, "Missing metadata key"

    # Stochastic-specific keys
    assert "signature_matrices" in result, "Missing signature_matrices key"
    assert "layer_info" in result, "Missing layer_info key"
    assert "encoder" in result, "Missing encoder key"

    # Verify embeddings are 1D vectors of correct dimension
    embeddings = result["embeddings"]
    assert len(embeddings) == G.number_of_nodes(), "Should have embedding for each node"

    for node, emb in embeddings.items():
        assert isinstance(emb, np.ndarray), f"Embedding for node {node} should be numpy array"
        assert emb.ndim == 1, f"Embedding for node {node} should be 1D"
        assert emb.shape == (embedding_dim,), f"Embedding for node {node} should have dim {embedding_dim}"

    # Verify signature matrices are still 2D
    matrices = result["signature_matrices"]
    assert len(matrices) == G.number_of_nodes(), "Should have matrix for each node"

    for node, matrix in matrices.items():
        assert isinstance(matrix, np.ndarray), f"Matrix for node {node} should be numpy array"
        assert matrix.ndim == 2, f"Matrix for node {node} should be 2D"
        assert matrix.shape[0] == matrix.shape[1], f"Matrix for node {node} should be square"

    # Verify layer info
    layer_info = result["layer_info"]
    for node, info in layer_info.items():
        assert "layers" in info, f"Missing layers for node {node}"
        assert "layer_sizes" in info, f"Missing layer_sizes for node {node}"
        assert "total_nodes" in info, f"Missing total_nodes for node {node}"
        assert info["total_nodes"] == matrices[node].shape[0], \
            f"total_nodes should match matrix dimension for node {node}"

    print(f"Computed {len(embeddings)} embeddings of shape ({embedding_dim},)")
    print(f"Example signature matrix shape for node 0: {matrices[0].shape}")
    print("Test 1 passed!")


def test_small_custom_graph():
    """Test 2: Small custom graph."""
    print("\n" + "=" * 50)
    print("Test 2: Small custom graph")
    print("-" * 50)

    G_small = nx.Graph()
    G_small.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])

    embedder = LSME(max_hops=2, n_samples=20, encoder_epochs=20, verbose=False)
    result = embedder.fit_transform(G_small)

    # Check layer structure for node 0
    info_0 = result["layer_info"][0]
    assert info_0["layers"][0] == [0], "Layer 0 should contain only root"
    assert set(info_0["layers"][1]) == {1, 2}, "Layer 1 should be neighbors of 0"

    # Check embeddings are 1D
    for node, emb in result["embeddings"].items():
        assert emb.ndim == 1, f"Embedding for node {node} should be 1D"

    print(f"Node 0 layers: {info_0['layers']}")
    print(f"Node 0 layer sizes: {info_0['layer_sizes']}")
    print("Test 2 passed!")


def test_isolated_node():
    """Test 3: Graph with isolated node."""
    print("\n" + "=" * 50)
    print("Test 3: Isolated node handling")
    print("-" * 50)

    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(1, 2)  # Node 0 is isolated

    embedder = LSME(max_hops=2, n_samples=10, encoder_epochs=20, verbose=False)
    result = embedder.fit_transform(G)

    # Isolated node should have 1x1 zero signature matrix
    assert result["signature_matrices"][0].shape == (1, 1), "Isolated node should have 1x1 matrix"
    assert result["signature_matrices"][0][0, 0] == 0.0, "Isolated node matrix should be zero"

    # But embedding should still be 1D with correct dimension
    assert result["embeddings"][0].ndim == 1, "Isolated node embedding should be 1D"

    print("Isolated node signature matrix:", result["signature_matrices"][0])
    print("Isolated node embedding shape:", result["embeddings"][0].shape)
    print("Test 3 passed!")


def test_reproducibility():
    """Test 4: Reproducibility with random_state."""
    print("\n" + "=" * 50)
    print("Test 4: Reproducibility")
    print("-" * 50)

    G = nx.karate_club_graph()

    embedder1 = LSME(max_hops=2, n_samples=50, encoder_epochs=20, verbose=False, random_state=42)
    result1 = embedder1.fit_transform(G)

    embedder2 = LSME(max_hops=2, n_samples=50, encoder_epochs=20, verbose=False, random_state=42)
    result2 = embedder2.fit_transform(G)

    # Signature matrices should be identical
    for node in G.nodes():
        assert np.allclose(result1["signature_matrices"][node],
                           result2["signature_matrices"][node]), \
            f"Matrices for node {node} should be identical with same random_state"

    # Embeddings should also be identical with same random_state
    for node in G.nodes():
        assert np.allclose(result1["embeddings"][node],
                           result2["embeddings"][node], atol=1e-5), \
            f"Embeddings for node {node} should be identical with same random_state"

    print("Results are reproducible with same random_state")
    print("Test 4 passed!")


def test_empty_graph():
    """Test 5: Empty graph."""
    print("\n" + "=" * 50)
    print("Test 5: Empty graph")
    print("-" * 50)

    G = nx.Graph()
    embedder = LSME(max_hops=2, n_samples=10, verbose=False)
    result = embedder.fit_transform(G)

    assert result["embeddings"] == {}, "Empty graph should return empty embeddings"
    assert result["signature_matrices"] == {}, "Empty graph should return empty matrices"
    assert result["layer_info"] == {}, "Empty graph should return empty layer_info"

    print("Empty graph handled correctly")
    print("Test 5 passed!")


def test_method_parameter():
    """Test 6: Method parameter works correctly."""
    print("\n" + "=" * 50)
    print("Test 6: Method parameter")
    print("-" * 50)

    G = nx.karate_club_graph()

    # Test each method
    for method in LSME.available_methods():
        embedder = LSME(method=method, max_hops=2, encoder_epochs=20, verbose=False)
        result = embedder.fit_transform(G)

        assert result['method'] == method, f"Method should be {method}"
        assert len(result['embeddings']) == G.number_of_nodes(), \
            f"{method}: Should have embedding per node"

        # All methods should produce 1D embeddings
        for node, emb in result['embeddings'].items():
            assert emb.ndim == 1, f"{method}: Embedding for node {node} should be 1D"

        print(f"  {method}: OK (embedding shape: {result['embeddings'][0].shape})")

    print("All methods work correctly")
    print("Test 6 passed!")


def test_available_methods():
    """Test 7: available_methods() static method."""
    print("\n" + "=" * 50)
    print("Test 7: available_methods()")
    print("-" * 50)

    methods = LSME.available_methods()

    assert isinstance(methods, list), "Should return a list"
    assert len(methods) >= 4, "Should have at least 4 methods"

    expected = ['stochastic', 'deterministic', 'random_walk', 'eigenvalue']
    for m in expected:
        assert m in methods, f"Method '{m}' should be available"

    print(f"Available methods: {methods}")
    print("Test 7 passed!")


def test_transform_alias():
    """Test 8: transform() is alias for fit_transform()."""
    print("\n" + "=" * 50)
    print("Test 8: transform() alias")
    print("-" * 50)

    G = nx.karate_club_graph()

    embedder = LSME(max_hops=2, n_samples=20, encoder_epochs=20, verbose=False, random_state=42)
    result1 = embedder.fit_transform(G)

    embedder2 = LSME(max_hops=2, n_samples=20, encoder_epochs=20, verbose=False, random_state=42)
    result2 = embedder2.transform(G)

    for node in G.nodes():
        assert np.allclose(result1['embeddings'][node], result2['embeddings'][node], atol=1e-5), \
            f"transform() should produce same result as fit_transform()"

    print("transform() is alias for fit_transform()")
    print("Test 8 passed!")


if __name__ == "__main__":
    test_basic_usage()
    test_small_custom_graph()
    test_isolated_node()
    test_reproducibility()
    test_empty_graph()
    test_method_parameter()
    test_available_methods()
    test_transform_alias()
    print("\n" + "=" * 50)
    print("All tests passed successfully!")
