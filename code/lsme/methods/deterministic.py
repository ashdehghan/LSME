"""Deterministic transition probability embedding method."""

import numpy as np
import networkx as nx
from typing import Any, Dict, List, Optional

from .base import BaseMethod
from ..core import get_nodes_by_hop_distance


class DeterministicMethod(BaseMethod):
    """
    Deterministic transition probability embedding method.

    For each node, computes the probability of edges connecting to
    previous, current, and next layers based on the BFS-organized
    neighborhood structure.

    Parameters
    ----------
    max_hops : int, default=2
        Maximum hop distance to consider for local neighborhoods.
    random_state : int, optional
        Random seed for reproducibility (not used in this deterministic method,
        but included for API consistency).
    """

    def __init__(
        self,
        max_hops: int = 2,
        random_state: Optional[int] = None
    ):
        super().__init__(max_hops=max_hops, random_state=random_state)

    @property
    def method_name(self) -> str:
        return "deterministic"

    def compute_node(self, G: nx.Graph, node: Any) -> np.ndarray:
        """
        Compute transition probability embedding for a single node.

        For each layer in the neighborhood:
        - pp: probability of edges to previous layer
        - pc: probability of edges within current layer
        - pn: probability of edges to next layer

        Parameters
        ----------
        G : networkx.Graph
            The input graph.
        node : Any
            The node to compute embedding for.

        Returns
        -------
        np.ndarray
            1D vector of shape (3 * (max_hops + 1),) containing
            [pp_0, pc_0, pn_0, pp_1, pc_1, pn_1, ...] for each layer.
        """
        layers = get_nodes_by_hop_distance(G, node, self.max_hops)

        # Build node-to-layer mapping
        node_to_layer: Dict[Any, int] = {}
        for hop, nodes in layers.items():
            for n in nodes:
                node_to_layer[n] = hop

        probs = []
        for hop in range(self.max_hops + 1):
            layer_nodes = layers.get(hop, [])

            if not layer_nodes:
                # No nodes at this layer
                probs.extend([0.0, 0.0, 0.0])
                continue

            ne_p = 0  # edges to previous layer
            ne_c = 0  # edges within current layer
            ne_n = 0  # edges to next layer

            for n in layer_nodes:
                for neighbor in G.neighbors(n):
                    neighbor_layer = node_to_layer.get(neighbor)
                    if neighbor_layer is None:
                        continue
                    if neighbor_layer < hop:
                        ne_p += 1
                    elif neighbor_layer == hop:
                        ne_c += 1
                    else:
                        ne_n += 1

            # Within-layer edges are counted twice (once from each endpoint)
            ne_c = ne_c // 2

            total = ne_p + ne_c + ne_n
            if total > 0:
                pp = ne_p / total
                pc = ne_c / total
                pn = ne_n / total
            else:
                pp = pc = pn = 0.0

            probs.extend([pp, pc, pn])

        return np.array(probs, dtype=np.float32)

    def compute(self, G: nx.Graph, verbose: bool = True) -> Dict[str, Any]:
        """
        Compute transition probability embeddings for all nodes.

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
            - "embeddings": dict mapping node_id to 1D probability vector
            - "method": "deterministic"
            - "params": algorithm parameters
            - "metadata": dict with layer info per node
        """
        if not isinstance(G, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")

        if G.number_of_nodes() == 0:
            return {
                "embeddings": {},
                "method": self.method_name,
                "params": self._get_params(),
                "metadata": {},
            }

        embeddings = {}
        metadata = {}
        node_list = list(G.nodes())

        for i, node in enumerate(node_list):
            if verbose:
                print(f"Computing embedding for node {node} ({i+1}/{len(node_list)})...")

            embedding = self.compute_node(G, node)
            embeddings[node] = embedding

            # Compute layer info for metadata
            layers = get_nodes_by_hop_distance(G, node, self.max_hops)
            layer_sizes = [len(layers.get(hop, [])) for hop in range(self.max_hops + 1)]
            metadata[node] = {
                "layer_sizes": layer_sizes,
                "total_nodes": sum(layer_sizes),
                "max_hop_reached": max(layers.keys()) if layers else 0,
            }

        if verbose:
            print(f"Computed {len(embeddings)} embeddings")

        return {
            "embeddings": embeddings,
            "method": self.method_name,
            "params": self._get_params(),
            "metadata": metadata,
        }

    def _get_params(self) -> Dict[str, Any]:
        """Return all parameters for this method."""
        return self._get_base_params()
