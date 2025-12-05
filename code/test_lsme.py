"""Test script to verify LSME package functionality."""

import networkx as nx
import numpy as np
from lsme import LSME


def test_basic_usage():
    """Test 1: Basic usage with Karate Club graph."""
    print("Test 1: Basic usage")
    print("-" * 50)

    G = nx.karate_club_graph()
    print(f"Graph: Karate Club")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    embedder = LSME(max_hops=2, n_samples=50, verbose=False)
    result = embedder.fit_transform(G)

    # Verify return structure
    assert isinstance(result, dict), "Result should be a dict"
    assert "signature_matrices" in result, "Missing signature_matrices key"
    assert "layer_info" in result, "Missing layer_info key"
    assert "params" in result, "Missing params key"

    # Verify signature matrices
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

    print(f"Computed {len(matrices)} signature matrices")
    print(f"Example matrix shape for node 0: {matrices[0].shape}")
    print("Test 1 passed!")


def test_small_custom_graph():
    """Test 2: Small custom graph."""
    print("\n" + "=" * 50)
    print("Test 2: Small custom graph")
    print("-" * 50)

    G_small = nx.Graph()
    G_small.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])

    embedder = LSME(max_hops=2, n_samples=20, verbose=False)
    result = embedder.fit_transform(G_small)

    # Check layer structure for node 0
    info_0 = result["layer_info"][0]
    assert info_0["layers"][0] == [0], "Layer 0 should contain only root"
    assert set(info_0["layers"][1]) == {1, 2}, "Layer 1 should be neighbors of 0"

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

    embedder = LSME(max_hops=2, n_samples=10, verbose=False)
    result = embedder.fit_transform(G)

    # Isolated node should have 1x1 zero matrix
    assert result["signature_matrices"][0].shape == (1, 1), "Isolated node should have 1x1 matrix"
    assert result["signature_matrices"][0][0, 0] == 0.0, "Isolated node matrix should be zero"

    print("Isolated node matrix:", result["signature_matrices"][0])
    print("Test 3 passed!")


def test_reproducibility():
    """Test 4: Reproducibility with random_state."""
    print("\n" + "=" * 50)
    print("Test 4: Reproducibility")
    print("-" * 50)

    G = nx.karate_club_graph()

    embedder1 = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result1 = embedder1.fit_transform(G)

    embedder2 = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    result2 = embedder2.fit_transform(G)

    for node in G.nodes():
        assert np.allclose(result1["signature_matrices"][node],
                           result2["signature_matrices"][node]), \
            f"Matrices for node {node} should be identical with same random_state"

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

    assert result["signature_matrices"] == {}, "Empty graph should return empty matrices"
    assert result["layer_info"] == {}, "Empty graph should return empty layer_info"

    print("Empty graph handled correctly")
    print("Test 5 passed!")


if __name__ == "__main__":
    test_basic_usage()
    test_small_custom_graph()
    test_isolated_node()
    test_reproducibility()
    test_empty_graph()
    print("\n" + "=" * 50)
    print("All tests passed successfully!")
