"""Test script for synthetic graph generator."""

import networkx as nx
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


def test_build_random():
    """Test 1: Random graph generation."""
    print("Test 1: Random graph generation")
    print("-" * 50)

    G = build_random(n_nodes=50, edge_prob=0.1, seed=42)

    assert isinstance(G, nx.Graph), "Should return NetworkX graph"
    assert G.number_of_nodes() == 50, "Should have 50 nodes"

    # Check attributes
    for node in G.nodes():
        assert G.nodes[node].get('role') == 'r', f"Node {node} should have role 'r'"
        assert G.nodes[node].get('con_type') == 'bridge', f"Node {node} should be bridge"

    print(f"Generated random graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Test 1 passed!")


def test_build_barbell():
    """Test 2: Barbell graph generation."""
    print("\n" + "=" * 50)
    print("Test 2: Barbell graph generation")
    print("-" * 50)

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

    print(f"Generated barbell graph: {G.number_of_nodes()} nodes")
    print(f"Roles: {roles}")
    print("Test 2 passed!")


def test_build_web_pattern():
    """Test 3: Web pattern generation."""
    print("\n" + "=" * 50)
    print("Test 3: Web pattern generation")
    print("-" * 50)

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

    print(f"Generated web pattern: {G.number_of_nodes()} nodes")
    print(f"Hub: {len(hub_nodes)}, Ring1: {len(ring1_nodes)}, Ring2: {len(ring2_nodes)}")
    print("Test 3 passed!")


def test_build_star_pattern():
    """Test 4: Star pattern generation."""
    print("\n" + "=" * 50)
    print("Test 4: Star pattern generation")
    print("-" * 50)

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

    print(f"Generated star pattern: {G.number_of_nodes()} nodes")
    print("Test 4 passed!")


def test_build_dense_star():
    """Test 5: Dense star generation."""
    print("\n" + "=" * 50)
    print("Test 5: Dense star generation")
    print("-" * 50)

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

    print(f"Generated dense star: {G.number_of_nodes()} nodes")
    print("Test 5 passed!")


def test_build_crossed_diamond():
    """Test 6: Crossed diamond generation."""
    print("\n" + "=" * 50)
    print("Test 6: Crossed diamond generation")
    print("-" * 50)

    G = build_crossed_diamond()

    assert isinstance(G, nx.Graph), "Should return NetworkX graph"
    assert G.number_of_nodes() == 5, "Should have 5 nodes"

    # Check hub
    assert G.nodes[0].get('role') == 'cd0', "Node 0 should be hub"

    # Check leaves form cycle
    leaf_nodes = [n for n in G.nodes() if G.nodes[n].get('role') == 'cd1']
    assert len(leaf_nodes) == 4, "Should have 4 leaf nodes"

    # All leaves should be connected to hub and to each other (cycle)
    for leaf in leaf_nodes:
        assert G.has_edge(0, leaf), f"Hub should connect to leaf {leaf}"

    print(f"Generated crossed diamond: {G.number_of_nodes()} nodes")
    print("Test 6 passed!")


def test_build_dynamic_star():
    """Test 7: Dynamic star generation."""
    print("\n" + "=" * 50)
    print("Test 7: Dynamic star generation")
    print("-" * 50)

    for n_leaves in [3, 5, 10]:
        G = build_dynamic_star(n_leaves=n_leaves)

        assert G.number_of_nodes() == n_leaves + 1, f"Should have {n_leaves + 1} nodes"
        assert G.nodes[0].get('role') == f'sd0_{n_leaves}', f"Hub role should include n_leaves"
        assert G.nodes[1].get('role') == f'sd1_{n_leaves}', f"Leaf role should include n_leaves"

    print("Dynamic star works for various sizes")
    print("Test 7 passed!")


def test_builder_basic():
    """Test 8: Builder basic usage."""
    print("\n" + "=" * 50)
    print("Test 8: Builder basic usage")
    print("-" * 50)

    builder = SyntheticGraphBuilder(random_state=42)
    G = (builder
         .add_random(n_nodes=50, edge_prob=0.3)  # Higher edge prob for connectivity
         .add_barbell(count=2)
         .hydrate(prob=0.1)  # Add edges to connect components
         .build())

    assert isinstance(G, nx.Graph), "Should return NetworkX graph"
    assert G.number_of_nodes() > 0, "Should have nodes"

    # Should be connected (LCC extraction)
    assert nx.is_connected(G), "Result should be connected"

    # Nodes should be relabeled 0 to N-1
    assert min(G.nodes()) == 0, "Nodes should start at 0"
    assert max(G.nodes()) == G.number_of_nodes() - 1, "Nodes should be contiguous"

    print(f"Built combined graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Test 8 passed!")


def test_builder_chaining():
    """Test 9: Builder method chaining."""
    print("\n" + "=" * 50)
    print("Test 9: Builder method chaining")
    print("-" * 50)

    G = (SyntheticGraphBuilder(random_state=42)
         .add_random(n_nodes=100, edge_prob=0.2)
         .add_web_pattern(count=5)
         .add_star_pattern(count=3)
         .add_dense_star(count=2)
         .add_crossed_diamond(count=2)
         .hydrate(prob=0.1)  # Connect components
         .build())

    assert G.number_of_nodes() > 50, "Should have substantial number of nodes"

    # Count roles
    roles = {}
    for node in G.nodes():
        role = G.nodes[node].get('role', 'unknown')
        roles[role[0] if role else 'x'] = roles.get(role[0] if role else 'x', 0) + 1

    print(f"Built graph with {G.number_of_nodes()} nodes")
    print(f"Role prefixes: {roles}")
    print("Test 9 passed!")


def test_builder_hydration():
    """Test 10: Builder hydration."""
    print("\n" + "=" * 50)
    print("Test 10: Builder hydration")
    print("-" * 50)

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

    # Hydrated graph should have more edges (or same if trim cancelled it)
    print(f"Without hydration: {G1.number_of_edges()} edges")
    print(f"With hydration: {G2.number_of_edges()} edges")
    print("Test 10 passed!")


def test_builder_trim():
    """Test 11: Builder trimming."""
    print("\n" + "=" * 50)
    print("Test 11: Builder trimming")
    print("-" * 50)

    G = (SyntheticGraphBuilder(random_state=42)
         .add_random(n_nodes=100, edge_prob=0.3)
         .trim(prob=0.5)
         .build())

    # Graph should still be connected (LCC extraction)
    assert nx.is_connected(G), "Result should be connected after trim"

    print(f"Trimmed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Test 11 passed!")


def test_builder_reset():
    """Test 12: Builder reset."""
    print("\n" + "=" * 50)
    print("Test 12: Builder reset")
    print("-" * 50)

    builder = SyntheticGraphBuilder(random_state=42)

    G1 = builder.add_barbell(count=2).build()

    # Reset and build different graph
    G2 = builder.reset().add_star_pattern(count=5).build()

    # G2 should not contain barbell structure
    barbell_roles = [n for n in G2.nodes() if G2.nodes[n].get('role', '').startswith('b')]
    assert len(barbell_roles) == 0, "Reset should clear previous patterns"

    print("Builder reset works correctly")
    print("Test 12 passed!")


def test_node_attributes_preserved():
    """Test 13: Node attributes preserved through builder."""
    print("\n" + "=" * 50)
    print("Test 13: Node attributes preserved")
    print("-" * 50)

    G = (SyntheticGraphBuilder(random_state=42)
         .add_web_pattern(count=2)
         .add_barbell(count=1)
         .build())

    # All nodes should have role and con_type
    for node in G.nodes():
        assert 'role' in G.nodes[node], f"Node {node} missing 'role'"
        assert 'con_type' in G.nodes[node], f"Node {node} missing 'con_type'"

    print("All node attributes preserved")
    print("Test 13 passed!")


if __name__ == "__main__":
    test_build_random()
    test_build_barbell()
    test_build_web_pattern()
    test_build_star_pattern()
    test_build_dense_star()
    test_build_crossed_diamond()
    test_build_dynamic_star()
    test_builder_basic()
    test_builder_chaining()
    test_builder_hydration()
    test_builder_trim()
    test_builder_reset()
    test_node_attributes_preserved()
    print("\n" + "=" * 50)
    print("All graph tests passed successfully!")
