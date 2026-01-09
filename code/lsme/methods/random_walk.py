"""Random walk-based transition probability embedding method."""

import random
import numpy as np
import networkx as nx
from typing import Any, Dict, Optional

from .base import BaseMethod
from ..core import get_nodes_by_hop_distance


class RandomWalkMethod(BaseMethod):
    """
    Random walk-based transition probability embedding method.

    Performs multiple random walks from each node and tracks transitions
    between layers (backward, same, forward) to compute probability vectors.

    Parameters
    ----------
    max_hops : int, default=2
        Maximum hop distance to consider for local neighborhoods.
    rw_length : int, default=10
        Length of each random walk (number of steps).
    sample_size : int, default=100
        Number of random walks to perform from each node.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_hops: int = 2,
        rw_length: int = 10,
        sample_size: int = 100,
        random_state: Optional[int] = None
    ):
        super().__init__(max_hops=max_hops, random_state=random_state)
        self.rw_length = rw_length
        self.sample_size = sample_size

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    @property
    def method_name(self) -> str:
        return "random_walk"

    def compute_node(self, G: nx.Graph, node: Any) -> np.ndarray:
        """
        Compute random walk transition probability embedding for a single node.

        Performs sample_size random walks of length rw_length and tracks:
        - pb: probability of moving backward (to lower layer)
        - ps: probability of staying in same layer
        - pf: probability of moving forward (to higher layer)

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
            [pb_0, ps_0, pf_0, pb_1, ps_1, pf_1, ...] for each layer.
        """
        layers = get_nodes_by_hop_distance(G, node, self.max_hops)

        # Build node-to-layer mapping
        node_to_layer: Dict[Any, int] = {}
        for hop, nodes in layers.items():
            for n in nodes:
                node_to_layer[n] = hop

        # Initialize transition counters for each layer
        transitions = {hop: {'b': 0, 's': 0, 'f': 0} for hop in range(self.max_hops + 1)}

        # Perform random walks
        for _ in range(self.sample_size):
            current = node
            current_layer = 0

            for _ in range(self.rw_length):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break

                next_node = random.choice(neighbors)
                next_layer = node_to_layer.get(next_node)

                # If next node is outside the neighborhood, treat as max+1
                if next_layer is None:
                    next_layer = self.max_hops + 1

                # Classify and count transition
                if current_layer <= self.max_hops:
                    if next_layer < current_layer:
                        transitions[current_layer]['b'] += 1
                    elif next_layer == current_layer:
                        transitions[current_layer]['s'] += 1
                    else:
                        transitions[current_layer]['f'] += 1

                current = next_node
                current_layer = min(next_layer, self.max_hops + 1)

        # Convert counts to probabilities
        probs = []
        for hop in range(self.max_hops + 1):
            total = transitions[hop]['b'] + transitions[hop]['s'] + transitions[hop]['f']
            if total > 0:
                pb = transitions[hop]['b'] / total
                ps = transitions[hop]['s'] / total
                pf = transitions[hop]['f'] / total
            else:
                pb = ps = pf = 0.0
            probs.extend([pb, ps, pf])

        return np.array(probs, dtype=np.float32)

    def compute(self, G: nx.Graph, verbose: bool = True) -> Dict[str, Any]:
        """
        Compute random walk embeddings for all nodes.

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
            - "method": "random_walk"
            - "params": algorithm parameters
            - "metadata": dict with layer info per node
        """
        if not isinstance(G, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")

        if G.is_directed():
            raise ValueError(
                "Random walk method requires an undirected graph. "
                "The transition probability calculation assumes bidirectional edges. "
                "Convert your graph using G.to_undirected() or use a different method."
            )

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
        params = self._get_base_params()
        params["rw_length"] = self.rw_length
        params["sample_size"] = self.sample_size
        return params
