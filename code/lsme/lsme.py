"""Main LSME class for computing local signature matrices."""

import random
import numpy as np
import networkx as nx
from typing import Optional
from .core import compute_local_signature_matrix


class LSME:
    """
    Local Structural Matrix Embeddings for graphs.

    Computes local signature matrices for nodes in a graph by averaging
    permuted local adjacency matrices.

    Parameters
    ----------
    max_hops : int, default=2
        Maximum hop distance to consider for local neighborhoods.

    n_samples : int, default=100
        Number of permutation samples to average for each signature matrix.

    verbose : bool, default=True
        Whether to print progress information.

    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_hops: int = 2,
        n_samples: int = 100,
        verbose: bool = True,
        random_state: Optional[int] = None
    ):
        self.max_hops = max_hops
        self.n_samples = n_samples
        self.verbose = verbose
        self.random_state = random_state

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def fit_transform(self, G: nx.Graph) -> dict:
        """
        Compute signature matrices for all nodes in the graph.

        Parameters
        ----------
        G : networkx.Graph
            The input graph.

        Returns
        -------
        dict
            Dictionary containing:
            - "signature_matrices": dict mapping node_id to 2D numpy array
            - "layer_info": dict mapping node_id to layer metadata
            - "params": dict with algorithm parameters used
        """
        if not isinstance(G, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")

        if G.number_of_nodes() == 0:
            return {
                "signature_matrices": {},
                "layer_info": {},
                "params": {"max_hops": self.max_hops, "n_samples": self.n_samples}
            }

        signature_matrices = {}
        layer_info = {}
        node_list = list(G.nodes())

        for i, node in enumerate(node_list):
            if self.verbose:
                print(f"Computing signature for node {node} ({i+1}/{len(node_list)})...")

            matrix, layers = compute_local_signature_matrix(
                G, node, self.max_hops, self.n_samples
            )

            signature_matrices[node] = matrix

            layer_sizes = [len(layers.get(hop, [])) for hop in range(self.max_hops + 1)]
            layer_info[node] = {
                "layers": layers,
                "layer_sizes": layer_sizes,
                "total_nodes": sum(layer_sizes),
                "max_hop_reached": max(layers.keys())
            }

        if self.verbose:
            print(f"Computed {len(signature_matrices)} signature matrices")

        return {
            "signature_matrices": signature_matrices,
            "layer_info": layer_info,
            "params": {"max_hops": self.max_hops, "n_samples": self.n_samples}
        }

    def transform(self, G: nx.Graph) -> dict:
        """
        Alias for fit_transform (no separate fitting needed for this algorithm).
        """
        return self.fit_transform(G)
