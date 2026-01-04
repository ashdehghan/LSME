"""Test script for LSME embedding methods."""

import networkx as nx
import numpy as np
from lsme import LSME


def test_stochastic_method():
    """Test 1: Stochastic method produces valid encoded embeddings and signature matrices."""
    print("Test 1: Stochastic method")
    print("-" * 50)

    G = nx.karate_club_graph()
    embedding_dim = 32
    lsme = LSME(method='stochastic', max_hops=2, n_samples=50, embedding_dim=embedding_dim,
                encoder_epochs=20, verbose=False, random_state=42)
    result = lsme.fit_transform(G)

    assert result['method'] == 'stochastic', "Method name should be 'stochastic'"
    assert 'embeddings' in result, "Result should have 'embeddings' key"
    assert 'signature_matrices' in result, "Stochastic should have 'signature_matrices'"
    assert 'layer_info' in result, "Stochastic should have 'layer_info'"
    assert 'encoder' in result, "Stochastic should have 'encoder'"

    # Embeddings should be 1D vectors (from encoder)
    for node, emb in result['embeddings'].items():
        assert isinstance(emb, np.ndarray), f"Embedding for {node} should be numpy array"
        assert emb.ndim == 1, f"Stochastic embedding for {node} should be 1D vector"
        assert emb.shape == (embedding_dim,), f"Embedding for {node} should have dim {embedding_dim}"

    # Signature matrices should still be 2D
    for node, matrix in result['signature_matrices'].items():
        assert isinstance(matrix, np.ndarray), f"Matrix for {node} should be numpy array"
        assert matrix.ndim == 2, f"Signature matrix for {node} should be 2D"
        assert matrix.shape[0] == matrix.shape[1], f"Matrix for {node} should be square"

    print(f"Generated {len(result['embeddings'])} embeddings of shape ({embedding_dim},)")
    print(f"Generated {len(result['signature_matrices'])} signature matrices")
    print("Test 1 passed!")


def test_deterministic_method():
    """Test 2: Deterministic method produces valid probability vectors."""
    print("\n" + "=" * 50)
    print("Test 2: Deterministic method")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(method='deterministic', max_hops=3, verbose=False)
    result = lsme.fit_transform(G)

    assert result['method'] == 'deterministic', "Method name should be 'deterministic'"
    assert 'embeddings' in result, "Result should have 'embeddings' key"
    assert 'metadata' in result, "Result should have 'metadata' key"

    expected_dim = 3 * (3 + 1)  # 3 * (max_hops + 1) = 12
    for node, vec in result['embeddings'].items():
        assert isinstance(vec, np.ndarray), f"Embedding for {node} should be numpy array"
        assert vec.ndim == 1, f"Deterministic embedding for {node} should be 1D vector"
        assert vec.shape[0] == expected_dim, f"Vector for {node} should have dim {expected_dim}"

        # Values should be probabilities (non-negative, sum can exceed 1 since we have triplets)
        assert np.all(vec >= 0), f"Probabilities for {node} should be non-negative"
        assert np.all(vec <= 1), f"Probabilities for {node} should be <= 1"

    print(f"Generated {len(result['embeddings'])} embeddings of shape ({expected_dim},)")
    print("Test 2 passed!")


def test_random_walk_method():
    """Test 3: Random walk method produces valid probability vectors."""
    print("\n" + "=" * 50)
    print("Test 3: Random walk method")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(
        method='random_walk',
        max_hops=2,
        rw_length=10,
        sample_size=50,
        verbose=False,
        random_state=42
    )
    result = lsme.fit_transform(G)

    assert result['method'] == 'random_walk', "Method name should be 'random_walk'"
    assert 'embeddings' in result, "Result should have 'embeddings' key"

    expected_dim = 3 * (2 + 1)  # 3 * (max_hops + 1) = 9
    for node, vec in result['embeddings'].items():
        assert isinstance(vec, np.ndarray), f"Embedding for {node} should be numpy array"
        assert vec.ndim == 1, f"Random walk embedding for {node} should be 1D vector"
        assert vec.shape[0] == expected_dim, f"Vector for {node} should have dim {expected_dim}"

        # Values should be probabilities
        assert np.all(vec >= 0), f"Probabilities for {node} should be non-negative"
        assert np.all(vec <= 1), f"Probabilities for {node} should be <= 1"

    print(f"Generated {len(result['embeddings'])} embeddings of shape ({expected_dim},)")
    print("Test 3 passed!")


def test_eigenvalue_method():
    """Test 4: Eigenvalue method produces valid eigenvalue vectors."""
    print("\n" + "=" * 50)
    print("Test 4: Eigenvalue method")
    print("-" * 50)

    G = nx.karate_club_graph()
    lsme = LSME(method='eigenvalue', max_hops=3, verbose=False)
    result = lsme.fit_transform(G)

    assert result['method'] == 'eigenvalue', "Method name should be 'eigenvalue'"
    assert 'embeddings' in result, "Result should have 'embeddings' key"

    expected_dim = 3 + 1  # max_hops + 1 = 4
    for node, vec in result['embeddings'].items():
        assert isinstance(vec, np.ndarray), f"Embedding for {node} should be numpy array"
        assert vec.ndim == 1, f"Eigenvalue embedding for {node} should be 1D vector"
        assert vec.shape[0] == expected_dim, f"Vector for {node} should have dim {expected_dim}"

    print(f"Generated {len(result['embeddings'])} embeddings of shape ({expected_dim},)")
    print("Test 4 passed!")


def test_unified_output_format():
    """Test 5: All methods return unified output format."""
    print("\n" + "=" * 50)
    print("Test 5: Unified output format")
    print("-" * 50)

    G = nx.karate_club_graph()
    methods = ['stochastic', 'deterministic', 'random_walk', 'eigenvalue']

    for method in methods:
        lsme = LSME(method=method, max_hops=2, encoder_epochs=20, verbose=False, random_state=42)
        result = lsme.fit_transform(G)

        # All methods should have these keys
        assert 'embeddings' in result, f"{method}: missing 'embeddings'"
        assert 'method' in result, f"{method}: missing 'method'"
        assert 'params' in result, f"{method}: missing 'params'"
        assert 'metadata' in result, f"{method}: missing 'metadata'"

        assert result['method'] == method, f"{method}: wrong method name"
        assert len(result['embeddings']) == G.number_of_nodes(), f"{method}: wrong embedding count"

        # All methods should produce 1D embeddings
        for node, emb in result['embeddings'].items():
            assert emb.ndim == 1, f"{method}: embedding for {node} should be 1D"

        print(f"  {method}: OK (embedding shape: {result['embeddings'][0].shape})")

    print("All methods return unified output format")
    print("Test 5 passed!")


def test_method_reproducibility():
    """Test 6: Random state produces reproducible results for all methods."""
    print("\n" + "=" * 50)
    print("Test 6: Method reproducibility")
    print("-" * 50)

    G = nx.karate_club_graph()
    methods = ['stochastic', 'random_walk']  # Only stochastic methods need testing

    for method in methods:
        lsme1 = LSME(method=method, max_hops=2, encoder_epochs=20, verbose=False, random_state=42)
        result1 = lsme1.fit_transform(G)

        lsme2 = LSME(method=method, max_hops=2, encoder_epochs=20, verbose=False, random_state=42)
        result2 = lsme2.fit_transform(G)

        for node in G.nodes():
            assert np.allclose(result1['embeddings'][node], result2['embeddings'][node], atol=1e-5), \
                f"{method}: embeddings for node {node} don't match"

        print(f"  {method}: reproducible")

    print("Stochastic methods are reproducible with same random_state")
    print("Test 6 passed!")


def test_invalid_method():
    """Test 7: Invalid method parameter raises ValueError."""
    print("\n" + "=" * 50)
    print("Test 7: Invalid method handling")
    print("-" * 50)

    try:
        lsme = LSME(method='invalid_method')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'invalid_method' in str(e), "Error should mention invalid method name"
        print(f"Correctly raised ValueError: {e}")

    print("Test 7 passed!")


def test_available_methods():
    """Test 8: available_methods() returns all methods."""
    print("\n" + "=" * 50)
    print("Test 8: available_methods() function")
    print("-" * 50)

    methods = LSME.available_methods()

    assert isinstance(methods, list), "available_methods() should return list"
    assert 'stochastic' in methods, "stochastic should be available"
    assert 'deterministic' in methods, "deterministic should be available"
    assert 'random_walk' in methods, "random_walk should be available"
    assert 'eigenvalue' in methods, "eigenvalue should be available"

    print(f"Available methods: {methods}")
    print("Test 8 passed!")


def test_empty_graph_all_methods():
    """Test 9: All methods handle empty graph correctly."""
    print("\n" + "=" * 50)
    print("Test 9: Empty graph handling")
    print("-" * 50)

    G = nx.Graph()
    methods = ['stochastic', 'deterministic', 'random_walk', 'eigenvalue']

    for method in methods:
        lsme = LSME(method=method, verbose=False)
        result = lsme.fit_transform(G)

        assert result['embeddings'] == {}, f"{method}: empty graph should return empty embeddings"
        print(f"  {method}: handles empty graph correctly")

    print("Test 9 passed!")


def test_isolated_node_all_methods():
    """Test 10: All methods handle isolated nodes."""
    print("\n" + "=" * 50)
    print("Test 10: Isolated node handling")
    print("-" * 50)

    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(1, 2)  # Node 0 is isolated

    methods = ['stochastic', 'deterministic', 'random_walk', 'eigenvalue']

    for method in methods:
        lsme = LSME(method=method, max_hops=2, encoder_epochs=20, verbose=False)
        result = lsme.fit_transform(G)

        assert 0 in result['embeddings'], f"{method}: should have embedding for isolated node"
        emb = result['embeddings'][0]
        assert isinstance(emb, np.ndarray), f"{method}: embedding should be numpy array"
        assert emb.ndim == 1, f"{method}: embedding should be 1D"
        print(f"  {method}: handles isolated node (shape={emb.shape})")

    print("Test 10 passed!")


if __name__ == "__main__":
    test_stochastic_method()
    test_deterministic_method()
    test_random_walk_method()
    test_eigenvalue_method()
    test_unified_output_format()
    test_method_reproducibility()
    test_invalid_method()
    test_available_methods()
    test_empty_graph_all_methods()
    test_isolated_node_all_methods()
    print("\n" + "=" * 50)
    print("All method tests passed successfully!")
