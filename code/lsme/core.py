"""Core shared utilities for Local Structural Matrix Embeddings."""

import networkx as nx
from collections import defaultdict
from typing import Any, Dict, List


def get_nodes_by_hop_distance(
    G: nx.Graph,
    root: Any,
    max_hops: int
) -> Dict[int, List[Any]]:
    """
    Get nodes organized by their hop distance from the root node.

    Uses BFS to explore the neighborhood and organize nodes into layers
    based on their shortest path distance from the root.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    root : Any
        The root node to start from.
    max_hops : int
        Maximum hop distance to explore.

    Returns
    -------
    dict
        Dictionary where keys are hop distances (0, 1, 2, ...)
        and values are lists of nodes at that distance.

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.path_graph(5)  # 0-1-2-3-4
    >>> layers = get_nodes_by_hop_distance(G, root=2, max_hops=2)
    >>> layers[0]  # Root node
    [2]
    >>> layers[1]  # 1 hop away
    [1, 3]
    >>> layers[2]  # 2 hops away
    [0, 4]
    """
    layers: Dict[int, List[Any]] = defaultdict(list)
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
