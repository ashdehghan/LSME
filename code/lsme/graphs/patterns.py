"""Individual graph pattern generators."""

import random
from typing import Optional

import networkx as nx


def build_random(
    n_nodes: int,
    edge_prob: float,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Build an Erdos-Renyi random graph.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    edge_prob : float
        Probability of edge between any two nodes.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        Random graph with node attributes:
        - role: "r" (random)
        - con_type: "bridge" (all nodes considered bridge nodes)
    """
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)

    # Set node attributes
    nx.set_node_attributes(G, "r", "role")
    nx.set_node_attributes(G, "bridge", "con_type")

    return G


def build_barbell(m1: int = 10, m2: int = 10) -> nx.Graph:
    """
    Build a barbell graph: two cliques connected by a path.

    Parameters
    ----------
    m1 : int, default=10
        Number of nodes in each clique.
    m2 : int, default=10
        Number of nodes in the connecting path.

    Returns
    -------
    nx.Graph
        Barbell graph with node attributes:
        - role: "b0" (clique members), "b1"-"b6" (path positions)
        - con_type: "non_bridge" or "bridge"
    """
    G = nx.barbell_graph(m1, m2)

    # Calculate node positions for role assignment
    n_total = 2 * m1 + m2

    for node in G.nodes():
        if node < m1 - 1:
            # Left clique (non-connector nodes)
            G.nodes[node]["role"] = "b0"
            G.nodes[node]["con_type"] = "non_bridge"
        elif node == m1 - 1:
            # Left connector to path
            G.nodes[node]["role"] = "b1"
            G.nodes[node]["con_type"] = "non_bridge"
        elif node >= m1 and node < m1 + m2:
            # Path nodes
            path_pos = node - m1
            if path_pos < m2 // 6:
                G.nodes[node]["role"] = "b2"
            elif path_pos < 2 * m2 // 6:
                G.nodes[node]["role"] = "b3"
            elif path_pos < 3 * m2 // 6:
                G.nodes[node]["role"] = "b4"
            elif path_pos < 4 * m2 // 6:
                G.nodes[node]["role"] = "b5"
            else:
                G.nodes[node]["role"] = "b6"
            G.nodes[node]["con_type"] = "bridge"
        elif node == m1 + m2:
            # Right connector from path
            G.nodes[node]["role"] = "b1"
            G.nodes[node]["con_type"] = "non_bridge"
        else:
            # Right clique (non-connector nodes)
            G.nodes[node]["role"] = "b0"
            G.nodes[node]["con_type"] = "non_bridge"

    return G


def build_web_pattern(n_rings: int = 2, spokes: int = 5) -> nx.Graph:
    """
    Build a web/spider pattern: hub with concentric rings and radial spokes.

    Parameters
    ----------
    n_rings : int, default=2
        Number of concentric rings around the hub.
    spokes : int, default=5
        Number of radial spokes.

    Returns
    -------
    nx.Graph
        Web pattern graph with node attributes:
        - role: "w0" (hub), "w1" (ring 1), "w2" (ring 2), etc.
        - con_type: "non_bridge" (inner), "bridge" (outer ring)
    """
    G = nx.Graph()

    # Add hub node
    hub = 0
    G.add_node(hub, role="w0", con_type="non_bridge")

    node_id = 1
    prev_ring_nodes = [hub] * spokes  # Virtual connections to hub for first ring

    for ring in range(1, n_rings + 1):
        ring_nodes = []
        nodes_in_ring = spokes * ring

        # Add nodes for this ring
        for i in range(nodes_in_ring):
            is_outer = ring == n_rings
            G.add_node(
                node_id,
                role=f"w{ring}",
                con_type="bridge" if is_outer else "non_bridge"
            )
            ring_nodes.append(node_id)
            node_id += 1

        # Connect within ring (circular)
        for i in range(nodes_in_ring):
            G.add_edge(ring_nodes[i], ring_nodes[(i + 1) % nodes_in_ring])

        # Connect to previous ring
        if ring == 1:
            # First ring connects directly to hub
            for node in ring_nodes:
                G.add_edge(hub, node)
        else:
            # Later rings connect to nodes in previous ring
            prev_size = len(prev_ring_nodes)
            curr_size = len(ring_nodes)
            ratio = curr_size / prev_size

            for i, node in enumerate(ring_nodes):
                # Connect to corresponding node in previous ring
                prev_idx = int(i / ratio)
                G.add_edge(node, prev_ring_nodes[prev_idx])

        prev_ring_nodes = ring_nodes

    return G


def build_star_pattern(n_arms: int = 5, arm_length: int = 2) -> nx.Graph:
    """
    Build a star pattern: central hub with radiating arms.

    Parameters
    ----------
    n_arms : int, default=5
        Number of arms extending from hub.
    arm_length : int, default=2
        Length of each arm (number of nodes per arm).

    Returns
    -------
    nx.Graph
        Star pattern graph with node attributes:
        - role: "s0" (hub), "s1" (first layer), "s2" (second layer), etc.
        - con_type: "non_bridge" (inner), "bridge" (tips)
    """
    G = nx.Graph()

    # Add hub
    hub = 0
    G.add_node(hub, role="s0", con_type="non_bridge")

    node_id = 1

    for arm in range(n_arms):
        prev_node = hub

        for depth in range(1, arm_length + 1):
            is_tip = depth == arm_length
            G.add_node(
                node_id,
                role=f"s{depth}",
                con_type="bridge" if is_tip else "non_bridge"
            )
            G.add_edge(prev_node, node_id)
            prev_node = node_id
            node_id += 1

    return G


def build_dense_star(n_leaves: int = 12) -> nx.Graph:
    """
    Build a dense star: central hub connected to all leaf nodes.

    Parameters
    ----------
    n_leaves : int, default=12
        Number of leaf nodes connected to the hub.

    Returns
    -------
    nx.Graph
        Dense star graph with node attributes:
        - role: "ds0" (hub), "ds1" (leaves)
        - con_type: "non_bridge" (hub), "bridge" (leaves)
    """
    G = nx.star_graph(n_leaves)

    # Hub is node 0 in star_graph
    G.nodes[0]["role"] = "ds0"
    G.nodes[0]["con_type"] = "non_bridge"

    for node in range(1, n_leaves + 1):
        G.nodes[node]["role"] = "ds1"
        G.nodes[node]["con_type"] = "bridge"

    return G


def build_crossed_diamond() -> nx.Graph:
    """
    Build a crossed diamond: hub with 4 leaves forming a complete cycle.

    The hub is connected to 4 leaf nodes, and all leaf nodes are
    connected to each other forming a cycle.

    Returns
    -------
    nx.Graph
        Crossed diamond graph with node attributes:
        - role: "cd0" (hub), "cd1" (leaves)
        - con_type: "non_bridge" (hub), "bridge" (leaves)
    """
    G = nx.Graph()

    # Add hub
    G.add_node(0, role="cd0", con_type="non_bridge")

    # Add 4 leaves
    for i in range(1, 5):
        G.add_node(i, role="cd1", con_type="bridge")
        G.add_edge(0, i)  # Connect to hub

    # Connect leaves in cycle
    for i in range(1, 5):
        G.add_edge(i, (i % 4) + 1)

    return G


def build_dynamic_star(n_leaves: int = 5) -> nx.Graph:
    """
    Build a parameterized star for testing pattern growth.

    Parameters
    ----------
    n_leaves : int, default=5
        Number of leaf nodes.

    Returns
    -------
    nx.Graph
        Dynamic star graph with node attributes:
        - role: "sd0_{n}" (hub), "sd1_{n}" (leaves) where n is leaf count
        - con_type: "non_bridge" (hub), "bridge" (leaves)
    """
    G = nx.star_graph(n_leaves)

    G.nodes[0]["role"] = f"sd0_{n_leaves}"
    G.nodes[0]["con_type"] = "non_bridge"

    for node in range(1, n_leaves + 1):
        G.nodes[node]["role"] = f"sd1_{n_leaves}"
        G.nodes[node]["con_type"] = "bridge"

    return G
