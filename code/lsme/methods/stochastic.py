"""Stochastic signature matrix embedding method."""

import random
import numpy as np
import networkx as nx
from typing import Any, Dict, List, Optional

from .base import BaseMethod
from ..core import get_nodes_by_hop_distance


def build_local_adjacency_matrix(
    G: nx.Graph,
    root: Any,
    layers: Dict[int, List[Any]],
    permute_layers: bool = True
) -> np.ndarray:
    """
    Build a local adjacency matrix centered at the root node.

    The matrix is organized by layers (hop distances):
    - Position (0,0) is the root node
    - Subsequent blocks contain nodes at increasing hop distances

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    root : Any
        The root node.
    layers : dict
        Dictionary mapping hop distance to list of nodes.
    permute_layers : bool, default=True
        If True, randomly shuffle nodes within each layer (except root).

    Returns
    -------
    np.ndarray
        The local adjacency matrix.
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


def compute_signature_matrix(
    G: nx.Graph,
    root: Any,
    max_hops: int = 2,
    n_samples: int = 100
) -> tuple[np.ndarray, Dict[int, List[Any]]]:
    """
    Compute the averaged local signature matrix for a given root node.

    This generates n_samples permutations of the local adjacency matrix
    (with shuffled ordering within layers) and returns their average.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    root : Any
        The root node.
    max_hops : int, default=2
        Maximum hop distance.
    n_samples : int, default=100
        Number of permutation samples.

    Returns
    -------
    tuple
        (signature_matrix, layers) where signature_matrix is the averaged
        adjacency matrix and layers is a dict mapping hop distance to node lists.
    """
    layers = get_nodes_by_hop_distance(G, root, max_hops)

    # Handle edge case: isolated node
    if len(layers) == 1 and len(layers[0]) == 1:
        return np.array([[0.0]], dtype=np.float32), layers

    # Generate first matrix to get dimensions
    first_matrix = build_local_adjacency_matrix(G, root, layers, permute_layers=False)
    accumulated_matrix = first_matrix.copy()

    # Generate and accumulate permuted matrices
    for _ in range(n_samples - 1):
        permuted_matrix = build_local_adjacency_matrix(G, root, layers, permute_layers=True)
        accumulated_matrix += permuted_matrix

    # Compute average
    avg_matrix = accumulated_matrix / n_samples

    return avg_matrix, layers


class StochasticMethod(BaseMethod):
    """
    Stochastic signature matrix embedding method.

    Computes local signature matrices by averaging multiple randomly
    permuted local adjacency matrices.

    Parameters
    ----------
    max_hops : int, default=2
        Maximum hop distance to consider for local neighborhoods.
    n_samples : int, default=100
        Number of permutation samples to average.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_hops: int = 2,
        n_samples: int = 100,
        random_state: Optional[int] = None
    ):
        super().__init__(max_hops=max_hops, random_state=random_state)
        self.n_samples = n_samples

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    @property
    def method_name(self) -> str:
        return "stochastic"

    def compute_node(self, G: nx.Graph, node: Any) -> np.ndarray:
        """
        Compute signature matrix for a single node.

        Parameters
        ----------
        G : networkx.Graph
            The input graph.
        node : Any
            The node to compute embedding for.

        Returns
        -------
        np.ndarray
            2D signature matrix for the node.
        """
        matrix, _ = compute_signature_matrix(G, node, self.max_hops, self.n_samples)
        return matrix

    def compute(self, G: nx.Graph, verbose: bool = True) -> Dict[str, Any]:
        """
        Compute signature matrices for all nodes in the graph.

        Parameters
        ----------
        G : networkx.Graph
            The input graph.
        verbose : bool, default=True
            Whether to print progress information.

        Returns
        -------
        dict
            Dictionary containing:
            - "embeddings": dict mapping node_id to 2D signature matrix
            - "method": "stochastic"
            - "params": algorithm parameters
            - "metadata": dict with layer info per node
            - "signature_matrices": same as embeddings (for compatibility)
            - "layer_info": detailed layer information per node
        """
        if not isinstance(G, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")

        if G.number_of_nodes() == 0:
            return {
                "embeddings": {},
                "method": self.method_name,
                "params": self._get_params(),
                "metadata": {},
                "signature_matrices": {},
                "layer_info": {},
            }

        embeddings = {}
        metadata = {}
        layer_info = {}
        node_list = list(G.nodes())

        for i, node in enumerate(node_list):
            if verbose:
                print(f"Computing signature for node {node} ({i+1}/{len(node_list)})...")

            matrix, layers = compute_signature_matrix(
                G, node, self.max_hops, self.n_samples
            )

            embeddings[node] = matrix

            layer_sizes = [len(layers.get(hop, [])) for hop in range(self.max_hops + 1)]
            node_metadata = {
                "layer_sizes": layer_sizes,
                "total_nodes": sum(layer_sizes),
                "max_hop_reached": max(layers.keys()),
            }
            metadata[node] = node_metadata
            layer_info[node] = {
                "layers": layers,
                **node_metadata,
            }

        if verbose:
            print(f"Computed {len(embeddings)} signature matrices")

        return {
            "embeddings": embeddings,
            "method": self.method_name,
            "params": self._get_params(),
            "metadata": metadata,
            "signature_matrices": embeddings,
            "layer_info": layer_info,
        }

    def _get_params(self) -> Dict[str, Any]:
        """Return all parameters for this method."""
        params = self._get_base_params()
        params["n_samples"] = self.n_samples
        return params
