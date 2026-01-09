"""Tests for LSME main class functionality."""

import numpy as np
import pytest
from lsme import LSME


class TestBasicUsage:
    """Test basic LSME usage with stochastic method."""

    def test_basic_usage(self, karate_graph):
        """Test basic usage with Karate Club graph (default stochastic method)."""
        embedding_dim = 32
        embedder = LSME(
            max_hops=2,
            n_samples=50,
            embedding_dim=embedding_dim,
            encoder_epochs=20,
            verbose=False
        )
        result = embedder.fit_transform(karate_graph)

        # Verify return structure (unified format)
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
        assert len(embeddings) == karate_graph.number_of_nodes(), \
            "Should have embedding for each node"

        for node, emb in embeddings.items():
            assert isinstance(emb, np.ndarray), \
                f"Embedding for node {node} should be numpy array"
            assert emb.ndim == 1, f"Embedding for node {node} should be 1D"
            assert emb.shape == (embedding_dim,), \
                f"Embedding for node {node} should have dim {embedding_dim}"

        # Verify signature matrices are still 2D
        matrices = result["signature_matrices"]
        assert len(matrices) == karate_graph.number_of_nodes(), \
            "Should have matrix for each node"

        for node, matrix in matrices.items():
            assert isinstance(matrix, np.ndarray), \
                f"Matrix for node {node} should be numpy array"
            assert matrix.ndim == 2, f"Matrix for node {node} should be 2D"
            assert matrix.shape[0] == matrix.shape[1], \
                f"Matrix for node {node} should be square"

        # Verify layer info
        layer_info = result["layer_info"]
        for node, info in layer_info.items():
            assert "layers" in info, f"Missing layers for node {node}"
            assert "layer_sizes" in info, f"Missing layer_sizes for node {node}"
            assert "total_nodes" in info, f"Missing total_nodes for node {node}"
            assert info["total_nodes"] == matrices[node].shape[0], \
                f"total_nodes should match matrix dimension for node {node}"

    def test_small_custom_graph(self, small_custom_graph):
        """Test with small custom graph."""
        embedder = LSME(max_hops=2, n_samples=20, encoder_epochs=20, verbose=False)
        result = embedder.fit_transform(small_custom_graph)

        # Check layer structure for node 0
        info_0 = result["layer_info"][0]
        assert info_0["layers"][0] == [0], "Layer 0 should contain only root"
        assert set(info_0["layers"][1]) == {1, 2}, "Layer 1 should be neighbors of 0"

        # Check embeddings are 1D
        for node, emb in result["embeddings"].items():
            assert emb.ndim == 1, f"Embedding for node {node} should be 1D"


class TestEdgeCases:
    """Test edge cases and special graph structures."""

    def test_isolated_node(self, isolated_node_graph):
        """Test graph with isolated node."""
        embedder = LSME(max_hops=2, n_samples=10, encoder_epochs=20, verbose=False)
        result = embedder.fit_transform(isolated_node_graph)

        # Isolated node should have 1x1 zero signature matrix
        assert result["signature_matrices"][0].shape == (1, 1), \
            "Isolated node should have 1x1 matrix"
        assert result["signature_matrices"][0][0, 0] == 0.0, \
            "Isolated node matrix should be zero"

        # But embedding should still be 1D with correct dimension
        assert result["embeddings"][0].ndim == 1, \
            "Isolated node embedding should be 1D"

    def test_empty_graph(self, empty_graph):
        """Test empty graph handling."""
        embedder = LSME(max_hops=2, n_samples=10, verbose=False)
        result = embedder.fit_transform(empty_graph)

        assert result["embeddings"] == {}, "Empty graph should return empty embeddings"
        assert result["signature_matrices"] == {}, \
            "Empty graph should return empty matrices"
        assert result["layer_info"] == {}, "Empty graph should return empty layer_info"

    def test_two_node_graph_stochastic_raises_error(self, two_node_graph):
        """Test stochastic method on 2-node graph raises error (requires >= 3 nodes)."""
        embedder = LSME(
            method='stochastic',
            max_hops=1,
            n_samples=10,
            encoder_epochs=10,
            verbose=False
        )
        with pytest.raises(ValueError, match="at least 3 nodes"):
            embedder.fit_transform(two_node_graph)

    def test_single_node_graph(self, single_node_graph):
        """Test single-node graph with all methods."""
        for method in ['deterministic', 'random_walk', 'eigenvalue']:
            embedder = LSME(method=method, max_hops=2, verbose=False)
            result = embedder.fit_transform(single_node_graph)

            assert len(result['embeddings']) == 1, f"{method}: Should have 1 embedding"
            assert result['embeddings'][0].ndim == 1, f"{method}: Embedding should be 1D"

    def test_disconnected_graph(self, disconnected_graph):
        """Test disconnected graph with multiple components."""
        embedder = LSME(method='deterministic', max_hops=2, verbose=False)
        result = embedder.fit_transform(disconnected_graph)

        # Should have embeddings for all 6 nodes
        assert len(result['embeddings']) == 6, "Should have 6 embeddings"

        # Each embedding should be valid
        for node, emb in result['embeddings'].items():
            assert emb.ndim == 1, f"Embedding for node {node} should be 1D"
            assert not np.any(np.isnan(emb)), f"Embedding for node {node} should not have NaN"

    def test_directed_graph_raises_error(self, directed_graph):
        """Test that directed graphs raise appropriate errors for ALL methods."""
        # All methods should raise error for directed graphs
        for method in ['stochastic', 'deterministic', 'random_walk', 'eigenvalue']:
            embedder = LSME(method=method, max_hops=2, verbose=False)
            with pytest.raises(ValueError, match="undirected"):
                embedder.fit_transform(directed_graph)

    def test_stochastic_single_node_raises_error(self, single_node_graph):
        """Test that stochastic method on single-node graph raises error (requires >= 3 nodes)."""
        embedder = LSME(
            method='stochastic',
            max_hops=1,
            n_samples=10,
            encoder_epochs=10,
            verbose=False
        )
        with pytest.raises(ValueError, match="at least 3 nodes"):
            embedder.fit_transform(single_node_graph)


class TestReproducibility:
    """Test reproducibility with random_state."""

    def test_reproducibility(self, karate_graph):
        """Test reproducibility with same random_state."""
        embedder1 = LSME(
            max_hops=2, n_samples=50, encoder_epochs=20,
            verbose=False, random_state=42
        )
        result1 = embedder1.fit_transform(karate_graph)

        embedder2 = LSME(
            max_hops=2, n_samples=50, encoder_epochs=20,
            verbose=False, random_state=42
        )
        result2 = embedder2.fit_transform(karate_graph)

        # Signature matrices should be identical
        for node in karate_graph.nodes():
            assert np.allclose(
                result1["signature_matrices"][node],
                result2["signature_matrices"][node]
            ), f"Matrices for node {node} should be identical with same random_state"

        # Embeddings should also be identical with same random_state
        for node in karate_graph.nodes():
            assert np.allclose(
                result1["embeddings"][node],
                result2["embeddings"][node],
                atol=1e-5
            ), f"Embeddings for node {node} should be identical with same random_state"


class TestMethods:
    """Test method parameter and available methods."""

    @pytest.mark.parametrize("method", ['stochastic', 'deterministic', 'random_walk', 'eigenvalue'])
    def test_method_parameter(self, karate_graph, method):
        """Test each method works correctly."""
        embedder = LSME(method=method, max_hops=2, encoder_epochs=20, verbose=False)
        result = embedder.fit_transform(karate_graph)

        assert result['method'] == method, f"Method should be {method}"
        assert len(result['embeddings']) == karate_graph.number_of_nodes(), \
            f"{method}: Should have embedding per node"

        # All methods should produce 1D embeddings
        for node, emb in result['embeddings'].items():
            assert emb.ndim == 1, f"{method}: Embedding for node {node} should be 1D"

    def test_available_methods(self):
        """Test available_methods() static method."""
        methods = LSME.available_methods()

        assert isinstance(methods, list), "Should return a list"
        assert len(methods) >= 4, "Should have at least 4 methods"

        expected = ['stochastic', 'deterministic', 'random_walk', 'eigenvalue']
        for m in expected:
            assert m in methods, f"Method '{m}' should be available"


class TestAliases:
    """Test method aliases."""

    def test_transform_alias(self, karate_graph):
        """Test transform() is alias for fit_transform()."""
        embedder = LSME(
            max_hops=2, n_samples=20, encoder_epochs=20,
            verbose=False, random_state=42
        )
        result1 = embedder.fit_transform(karate_graph)

        embedder2 = LSME(
            max_hops=2, n_samples=20, encoder_epochs=20,
            verbose=False, random_state=42
        )
        result2 = embedder2.transform(karate_graph)

        for node in karate_graph.nodes():
            assert np.allclose(
                result1['embeddings'][node],
                result2['embeddings'][node],
                atol=1e-5
            ), "transform() should produce same result as fit_transform()"
