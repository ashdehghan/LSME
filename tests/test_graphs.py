"""Tests for synthetic graph generator."""

import networkx as nx
import pytest
from lsme.graphs import (
    SyntheticGraphBuilder,
    build_random,
    build_barbell,
    build_web_pattern,
    build_star_pattern,
    build_dense_star,
    build_crossed_diamond,
    build_dynamic_star,
)


class TestBuildRandom:
    """Test random graph generation."""

    def test_build_random(self):
        """Test random graph generation."""
        G = build_random(n_nodes=50, edge_prob=0.1, seed=42)

        assert isinstance(G, nx.Graph), "Should return NetworkX graph"
        assert G.number_of_nodes() == 50, "Should have 50 nodes"

        # Check attributes
        for node in G.nodes():
            assert G.nodes[node].get('role') == 'r', f"Node {node} should have role 'r'"
            assert G.nodes[node].get('con_type') == 'bridge', f"Node {node} should be bridge"


class TestBuildBarbell:
    """Test barbell graph generation."""

    def test_build_barbell(self):
        """Test barbell graph generation."""
        G = build_barbell(m1=10, m2=10)

        assert isinstance(G, nx.Graph), "Should return NetworkX graph"
        expected_nodes = 2 * 10 + 10  # Two cliques + path
        assert G.number_of_nodes() == expected_nodes, f"Should have {expected_nodes} nodes"

        # Check attributes exist
        roles = set(G.nodes[n].get('role') for n in G.nodes())
        assert len(roles) > 1, "Should have multiple roles"

        con_types = set(G.nodes[n].get('con_type') for n in G.nodes())
        assert 'bridge' in con_types, "Should have bridge nodes"
        assert 'non_bridge' in con_types, "Should have non-bridge nodes"


class TestBuildWebPattern:
    """Test web pattern generation."""

    def test_build_web_pattern(self):
        """Test web pattern generation."""
        G = build_web_pattern(n_rings=2, spokes=5)

        assert isinstance(G, nx.Graph), "Should return NetworkX graph"
        assert G.number_of_nodes() > 1, "Should have multiple nodes"

        # Check hub exists
        hub_nodes = [n for n in G.nodes() if G.nodes[n].get('role') == 'w0']
        assert len(hub_nodes) == 1, "Should have exactly one hub (w0)"

        # Check rings
        ring1_nodes = [n for n in G.nodes() if G.nodes[n].get('role') == 'w1']
        ring2_nodes = [n for n in G.nodes() if G.nodes[n].get('role') == 'w2']
        assert len(ring1_nodes) > 0, "Should have ring 1 nodes"
        assert len(ring2_nodes) > 0, "Should have ring 2 nodes"


class TestBuildStarPattern:
    """Test star pattern generation."""

    def test_build_star_pattern(self):
        """Test star pattern generation."""
        G = build_star_pattern(n_arms=5, arm_length=3)

        assert isinstance(G, nx.Graph), "Should return NetworkX graph"
        expected_nodes = 1 + 5 * 3  # hub + arms * length
        assert G.number_of_nodes() == expected_nodes, f"Should have {expected_nodes} nodes"

        # Check structure
        hub_nodes = [n for n in G.nodes() if G.nodes[n].get('role') == 's0']
        assert len(hub_nodes) == 1, "Should have exactly one hub"

        # Tips should be bridges
        tip_nodes = [n for n in G.nodes() if G.nodes[n].get('role') == 's3']
        assert len(tip_nodes) == 5, "Should have 5 tips (arm ends)"
        for node in tip_nodes:
            assert G.nodes[node].get('con_type') == 'bridge', "Tips should be bridge nodes"


class TestBuildDenseStar:
    """Test dense star generation."""

    def test_build_dense_star(self):
        """Test dense star generation."""
        G = build_dense_star(n_leaves=12)

        assert isinstance(G, nx.Graph), "Should return NetworkX graph"
        assert G.number_of_nodes() == 13, "Should have 13 nodes (1 hub + 12 leaves)"

        # Check hub
        assert G.nodes[0].get('role') == 'ds0', "Node 0 should be hub"
        assert G.nodes[0].get('con_type') == 'non_bridge', "Hub should be non-bridge"

        # Check all leaves
        for node in range(1, 13):
            assert G.nodes[node].get('role') == 'ds1', f"Node {node} should be leaf"
            assert G.nodes[node].get('con_type') == 'bridge', f"Node {node} should be bridge"


class TestBuildCrossedDiamond:
    """Test crossed diamond generation."""

    def test_build_crossed_diamond(self):
        """Test crossed diamond generation."""
        G = build_crossed_diamond()

        assert isinstance(G, nx.Graph), "Should return NetworkX graph"
        assert G.number_of_nodes() == 5, "Should have 5 nodes"

        # Check hub
        assert G.nodes[0].get('role') == 'cd0', "Node 0 should be hub"

        # Check leaves form cycle
        leaf_nodes = [n for n in G.nodes() if G.nodes[n].get('role') == 'cd1']
        assert len(leaf_nodes) == 4, "Should have 4 leaf nodes"

        # All leaves should be connected to hub
        for leaf in leaf_nodes:
            assert G.has_edge(0, leaf), f"Hub should connect to leaf {leaf}"


class TestBuildDynamicStar:
    """Test dynamic star generation."""

    @pytest.mark.parametrize("n_leaves", [3, 5, 10])
    def test_build_dynamic_star(self, n_leaves):
        """Test dynamic star generation with various sizes."""
        G = build_dynamic_star(n_leaves=n_leaves)

        assert G.number_of_nodes() == n_leaves + 1, f"Should have {n_leaves + 1} nodes"
        assert G.nodes[0].get('role') == f'sd0_{n_leaves}', "Hub role should include n_leaves"
        assert G.nodes[1].get('role') == f'sd1_{n_leaves}', "Leaf role should include n_leaves"


class TestBuilderBasic:
    """Test builder basic usage."""

    def test_builder_basic(self):
        """Test builder basic usage."""
        builder = SyntheticGraphBuilder(random_state=42)
        G = (builder
             .add_random(n_nodes=50, edge_prob=0.3)
             .add_barbell(count=2)
             .hydrate(prob=0.1)
             .build())

        assert isinstance(G, nx.Graph), "Should return NetworkX graph"
        assert G.number_of_nodes() > 0, "Should have nodes"

        # Should be connected (LCC extraction)
        assert nx.is_connected(G), "Result should be connected"

        # Nodes should be relabeled 0 to N-1
        assert min(G.nodes()) == 0, "Nodes should start at 0"
        assert max(G.nodes()) == G.number_of_nodes() - 1, "Nodes should be contiguous"


class TestBuilderChaining:
    """Test builder method chaining."""

    def test_builder_chaining(self):
        """Test builder method chaining with multiple patterns."""
        G = (SyntheticGraphBuilder(random_state=42)
             .add_random(n_nodes=100, edge_prob=0.2)
             .add_web_pattern(count=5)
             .add_star_pattern(count=3)
             .add_dense_star(count=2)
             .add_crossed_diamond(count=2)
             .hydrate(prob=0.1)
             .build())

        assert G.number_of_nodes() > 50, "Should have substantial number of nodes"


class TestBuilderHydration:
    """Test builder hydration."""

    def test_builder_hydration(self):
        """Test builder hydration adds edges."""
        # Build without hydration
        G1 = (SyntheticGraphBuilder(random_state=42)
              .add_barbell(count=3)
              .add_star_pattern(count=3)
              .build())

        # Build with hydration
        G2 = (SyntheticGraphBuilder(random_state=42)
              .add_barbell(count=3)
              .add_star_pattern(count=3)
              .hydrate(prob=0.3)
              .build())

        # Both should be valid graphs
        assert isinstance(G1, nx.Graph)
        assert isinstance(G2, nx.Graph)


class TestBuilderTrim:
    """Test builder trimming."""

    def test_builder_trim(self):
        """Test builder trimming removes edges."""
        G = (SyntheticGraphBuilder(random_state=42)
             .add_random(n_nodes=100, edge_prob=0.3)
             .trim(prob=0.5)
             .build())

        # Graph should still be connected (LCC extraction)
        assert nx.is_connected(G), "Result should be connected after trim"


class TestBuilderReset:
    """Test builder reset."""

    def test_builder_reset(self):
        """Test builder reset clears state."""
        builder = SyntheticGraphBuilder(random_state=42)

        G1 = builder.add_barbell(count=2).build()

        # Reset and build different graph
        G2 = builder.reset().add_star_pattern(count=5).build()

        # G2 should not contain barbell structure
        barbell_roles = [n for n in G2.nodes() if G2.nodes[n].get('role', '').startswith('b')]
        assert len(barbell_roles) == 0, "Reset should clear previous patterns"


class TestNodeAttributes:
    """Test node attributes are preserved."""

    def test_node_attributes_preserved(self):
        """Test node attributes preserved through builder."""
        G = (SyntheticGraphBuilder(random_state=42)
             .add_web_pattern(count=2)
             .add_barbell(count=1)
             .build())

        # All nodes should have role and con_type
        for node in G.nodes():
            assert 'role' in G.nodes[node], f"Node {node} missing 'role'"
            assert 'con_type' in G.nodes[node], f"Node {node} missing 'con_type'"
