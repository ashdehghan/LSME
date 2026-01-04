"""Tests for LSME embedding methods."""

import numpy as np
import pytest
from lsme import LSME


class TestStochasticMethod:
    """Test stochastic method produces valid encoded embeddings."""

    def test_stochastic_method(self, karate_graph):
        """Test stochastic method produces valid encoded embeddings and signature matrices."""
        embedding_dim = 32
        lsme = LSME(
            method='stochastic',
            max_hops=2,
            n_samples=50,
            embedding_dim=embedding_dim,
            encoder_epochs=20,
            verbose=False,
            random_state=42
        )
        result = lsme.fit_transform(karate_graph)

        assert result['method'] == 'stochastic', "Method name should be 'stochastic'"
        assert 'embeddings' in result, "Result should have 'embeddings' key"
        assert 'signature_matrices' in result, "Stochastic should have 'signature_matrices'"
        assert 'layer_info' in result, "Stochastic should have 'layer_info'"
        assert 'encoder' in result, "Stochastic should have 'encoder'"

        # Embeddings should be 1D vectors (from encoder)
        for node, emb in result['embeddings'].items():
            assert isinstance(emb, np.ndarray), f"Embedding for {node} should be numpy array"
            assert emb.ndim == 1, f"Stochastic embedding for {node} should be 1D vector"
            assert emb.shape == (embedding_dim,), \
                f"Embedding for {node} should have dim {embedding_dim}"

        # Signature matrices should still be 2D
        for node, matrix in result['signature_matrices'].items():
            assert isinstance(matrix, np.ndarray), f"Matrix for {node} should be numpy array"
            assert matrix.ndim == 2, f"Signature matrix for {node} should be 2D"
            assert matrix.shape[0] == matrix.shape[1], f"Matrix for {node} should be square"


class TestDeterministicMethod:
    """Test deterministic method produces valid probability vectors."""

    def test_deterministic_method(self, karate_graph):
        """Test deterministic method produces valid probability vectors."""
        lsme = LSME(method='deterministic', max_hops=3, verbose=False)
        result = lsme.fit_transform(karate_graph)

        assert result['method'] == 'deterministic', "Method name should be 'deterministic'"
        assert 'embeddings' in result, "Result should have 'embeddings' key"
        assert 'metadata' in result, "Result should have 'metadata' key"

        expected_dim = 3 * (3 + 1)  # 3 * (max_hops + 1) = 12
        for node, vec in result['embeddings'].items():
            assert isinstance(vec, np.ndarray), f"Embedding for {node} should be numpy array"
            assert vec.ndim == 1, f"Deterministic embedding for {node} should be 1D vector"
            assert vec.shape[0] == expected_dim, \
                f"Vector for {node} should have dim {expected_dim}"

            # Values should be probabilities
            assert np.all(vec >= 0), f"Probabilities for {node} should be non-negative"
            assert np.all(vec <= 1), f"Probabilities for {node} should be <= 1"


class TestRandomWalkMethod:
    """Test random walk method produces valid probability vectors."""

    def test_random_walk_method(self, karate_graph):
        """Test random walk method produces valid probability vectors."""
        lsme = LSME(
            method='random_walk',
            max_hops=2,
            rw_length=10,
            sample_size=50,
            verbose=False,
            random_state=42
        )
        result = lsme.fit_transform(karate_graph)

        assert result['method'] == 'random_walk', "Method name should be 'random_walk'"
        assert 'embeddings' in result, "Result should have 'embeddings' key"

        expected_dim = 3 * (2 + 1)  # 3 * (max_hops + 1) = 9
        for node, vec in result['embeddings'].items():
            assert isinstance(vec, np.ndarray), f"Embedding for {node} should be numpy array"
            assert vec.ndim == 1, f"Random walk embedding for {node} should be 1D vector"
            assert vec.shape[0] == expected_dim, \
                f"Vector for {node} should have dim {expected_dim}"

            # Values should be probabilities
            assert np.all(vec >= 0), f"Probabilities for {node} should be non-negative"
            assert np.all(vec <= 1), f"Probabilities for {node} should be <= 1"


class TestEigenvalueMethod:
    """Test eigenvalue method produces valid eigenvalue vectors."""

    def test_eigenvalue_method(self, karate_graph):
        """Test eigenvalue method produces valid eigenvalue vectors."""
        lsme = LSME(method='eigenvalue', max_hops=3, verbose=False)
        result = lsme.fit_transform(karate_graph)

        assert result['method'] == 'eigenvalue', "Method name should be 'eigenvalue'"
        assert 'embeddings' in result, "Result should have 'embeddings' key"

        expected_dim = 3 + 1  # max_hops + 1 = 4
        for node, vec in result['embeddings'].items():
            assert isinstance(vec, np.ndarray), f"Embedding for {node} should be numpy array"
            assert vec.ndim == 1, f"Eigenvalue embedding for {node} should be 1D vector"
            assert vec.shape[0] == expected_dim, \
                f"Vector for {node} should have dim {expected_dim}"


class TestUnifiedOutputFormat:
    """Test all methods return unified output format."""

    @pytest.mark.parametrize("method", ['stochastic', 'deterministic', 'random_walk', 'eigenvalue'])
    def test_unified_output_format(self, karate_graph, method):
        """Test all methods return unified output format."""
        lsme = LSME(method=method, max_hops=2, encoder_epochs=20, verbose=False, random_state=42)
        result = lsme.fit_transform(karate_graph)

        # All methods should have these keys
        assert 'embeddings' in result, f"{method}: missing 'embeddings'"
        assert 'method' in result, f"{method}: missing 'method'"
        assert 'params' in result, f"{method}: missing 'params'"
        assert 'metadata' in result, f"{method}: missing 'metadata'"

        assert result['method'] == method, f"{method}: wrong method name"
        assert len(result['embeddings']) == karate_graph.number_of_nodes(), \
            f"{method}: wrong embedding count"

        # All methods should produce 1D embeddings
        for node, emb in result['embeddings'].items():
            assert emb.ndim == 1, f"{method}: embedding for {node} should be 1D"


class TestMethodReproducibility:
    """Test random state produces reproducible results."""

    @pytest.mark.parametrize("method", ['stochastic', 'random_walk'])
    def test_method_reproducibility(self, karate_graph, method):
        """Test stochastic methods are reproducible with same random_state."""
        lsme1 = LSME(
            method=method, max_hops=2, encoder_epochs=20,
            verbose=False, random_state=42
        )
        result1 = lsme1.fit_transform(karate_graph)

        lsme2 = LSME(
            method=method, max_hops=2, encoder_epochs=20,
            verbose=False, random_state=42
        )
        result2 = lsme2.fit_transform(karate_graph)

        for node in karate_graph.nodes():
            assert np.allclose(
                result1['embeddings'][node],
                result2['embeddings'][node],
                atol=1e-5
            ), f"{method}: embeddings for node {node} don't match"


class TestInvalidMethod:
    """Test invalid method handling."""

    def test_invalid_method(self):
        """Test invalid method parameter raises ValueError."""
        with pytest.raises(ValueError, match='invalid_method'):
            LSME(method='invalid_method')


class TestAvailableMethods:
    """Test available_methods() function."""

    def test_available_methods(self):
        """Test available_methods() returns all methods."""
        methods = LSME.available_methods()

        assert isinstance(methods, list), "available_methods() should return list"
        assert 'stochastic' in methods, "stochastic should be available"
        assert 'deterministic' in methods, "deterministic should be available"
        assert 'random_walk' in methods, "random_walk should be available"
        assert 'eigenvalue' in methods, "eigenvalue should be available"


class TestEmptyGraph:
    """Test empty graph handling."""

    @pytest.mark.parametrize("method", ['stochastic', 'deterministic', 'random_walk', 'eigenvalue'])
    def test_empty_graph_all_methods(self, empty_graph, method):
        """Test all methods handle empty graph correctly."""
        lsme = LSME(method=method, verbose=False)
        result = lsme.fit_transform(empty_graph)

        assert result['embeddings'] == {}, \
            f"{method}: empty graph should return empty embeddings"


class TestIsolatedNode:
    """Test isolated node handling."""

    @pytest.mark.parametrize("method", ['stochastic', 'deterministic', 'random_walk', 'eigenvalue'])
    def test_isolated_node_all_methods(self, isolated_node_graph, method):
        """Test all methods handle isolated nodes."""
        lsme = LSME(method=method, max_hops=2, encoder_epochs=20, verbose=False)
        result = lsme.fit_transform(isolated_node_graph)

        assert 0 in result['embeddings'], \
            f"{method}: should have embedding for isolated node"
        emb = result['embeddings'][0]
        assert isinstance(emb, np.ndarray), f"{method}: embedding should be numpy array"
        assert emb.ndim == 1, f"{method}: embedding should be 1D"
