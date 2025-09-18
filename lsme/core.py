"""Core algorithm functions for Local Structural Matrix Embeddings."""

import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List
import random


def get_nodes_by_hop_distance(G: nx.Graph, root: int, max_hops: int) -> Dict[int, List[int]]:
    """
    Get nodes organized by their hop distance from the root node.

    Returns a dict where keys are hop distances (0, 1, 2, ...)
    and values are lists of nodes at that distance.
    """
    layers = defaultdict(list)
    visited = {root}
    layers[0] = [root]

    current_layer = {root}

    for hop in range(1, max_hops + 1):
        next_layer = set()
        for node in current_layer:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    next_layer.add(neighbor)
                    visited.add(neighbor)

        if not next_layer:
            break

        layers[hop] = sorted(list(next_layer))
        current_layer = next_layer

    return dict(layers)


def build_local_adjacency_matrix(
    G: nx.Graph,
    root: int,
    layers: Dict[int, List[int]],
    permute_layers: bool = True
) -> np.ndarray:
    """
    Build a local adjacency matrix centered at the root node.

    The matrix is organized by layers (hop distances):
    - Position (0,0) is the root node
    - Subsequent blocks contain nodes at increasing hop distances

    If permute_layers is True, randomly shuffle nodes within each layer (except root).
    """
    node_order = []

    for hop in sorted(layers.keys()):
        layer_nodes = layers[hop].copy()

        # Permute nodes within layer (except root layer)
        if permute_layers and hop > 0:
            random.shuffle(layer_nodes)

        node_order.extend(layer_nodes)

    n = len(node_order)
    adj_matrix = np.zeros((n, n), dtype=np.float32)

    for i, node_i in enumerate(node_order):
        for j, node_j in enumerate(node_order):
            if G.has_edge(node_i, node_j):
                adj_matrix[i, j] = 1.0

    return adj_matrix


def compute_local_signature_matrix(
    G: nx.Graph,
    root: int,
    max_hops: int = 2,
    n_samples: int = 100
) -> np.ndarray:
    """
    Compute the averaged local signature matrix for a given root node.

    This generates n_samples permutations of the local adjacency matrix
    (with shuffled ordering within layers) and returns their average.
    """
    layers = get_nodes_by_hop_distance(G, root, max_hops)

    # Handle edge case: isolated node
    if len(layers) == 1 and len(layers[0]) == 1:
        return np.array([[0.0]])

    # Generate first matrix to get dimensions
    first_matrix = build_local_adjacency_matrix(G, root, layers, permute_layers=False)
    accumulated_matrix = first_matrix.copy()

    # Generate and accumulate permuted matrices
    for _ in range(n_samples - 1):
        permuted_matrix = build_local_adjacency_matrix(G, root, layers, permute_layers=True)
        accumulated_matrix += permuted_matrix

    # Compute average
    avg_matrix = accumulated_matrix / n_samples

    return avg_matrix