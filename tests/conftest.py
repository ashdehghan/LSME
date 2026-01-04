"""Shared pytest fixtures for LSME tests."""

import pytest
import networkx as nx


# ============================================================
# Graph Fixtures
# ============================================================

@pytest.fixture
def karate_graph():
    """Zachary's Karate Club graph - standard benchmark with 34 nodes."""
    return nx.karate_club_graph()


@pytest.fixture
def small_custom_graph():
    """Small 4-node graph: 0-1-2-3 with triangle 0-1-2."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])
    return G


@pytest.fixture
def isolated_node_graph():
    """3-node graph where node 0 is isolated, nodes 1-2 are connected."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(1, 2)
    return G


@pytest.fixture
def empty_graph():
    """Empty graph with no nodes or edges."""
    return nx.Graph()


@pytest.fixture
def star_with_extras():
    """Star graph (10 nodes) with extra edges for variable neighborhood sizes."""
    G = nx.star_graph(9)
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])
    return G


@pytest.fixture
def path_graph_3():
    """Simple 3-node path graph: 0-1-2."""
    return nx.path_graph(3)


# ============================================================
# LSME Result Fixtures (for encoder tests)
# ============================================================

@pytest.fixture
def lsme_result_karate(karate_graph):
    """Pre-computed LSME signature matrices for karate graph."""
    from lsme import LSME
    lsme = LSME(max_hops=2, n_samples=50, verbose=False, random_state=42)
    return lsme.fit_transform(karate_graph)


@pytest.fixture
def lsme_result_star(star_with_extras):
    """Pre-computed LSME signature matrices for star graph (variable sizes)."""
    from lsme import LSME
    lsme = LSME(max_hops=1, n_samples=30, verbose=False, random_state=42)
    return lsme.fit_transform(star_with_extras)


@pytest.fixture
def lsme_result_path(path_graph_3):
    """Pre-computed LSME signature matrices for path graph."""
    from lsme import LSME
    lsme = LSME(max_hops=2, n_samples=30, verbose=False, random_state=42)
    return lsme.fit_transform(path_graph_3)


# ============================================================
# Pytest Configuration
# ============================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (training-intensive)")
    config.addinivalue_line("markers", "gpu: marks tests requiring CUDA")
