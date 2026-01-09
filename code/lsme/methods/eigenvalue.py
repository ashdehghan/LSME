"""Eigenvalue-based embedding method using transition probability matrices."""

import warnings
import numpy as np
import networkx as nx
from typing import Any, Dict, Optional

from .base import BaseMethod
from ..core import get_nodes_by_hop_distance


class EigenvalueMethod(BaseMethod):
    """
    Eigenvalue-based embedding method.

    For each node, builds a transition probability matrix based on
    layer structure and extracts eigenvalues as the embedding.

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
        return "eigenvalue"

    def compute_node(self, G: nx.Graph, node: Any) -> np.ndarray:
        """
        Compute eigenvalue-based embedding for a single node.

        Builds a transition probability matrix based on layer structure
        and extracts sorted eigenvalues.

        Parameters
        ----------
        G : networkx.Graph
            The input graph.
        node : Any
            The node to compute embedding for.

        Returns
        -------
        np.ndarray
            1D vector of shape (max_hops + 1,) containing sorted eigenvalues.
        """
        layers = get_nodes_by_hop_distance(G, node, self.max_hops)
        n_layers = len(layers)

        # Handle edge case: isolated node or single layer
        if n_layers <= 1:
            result = np.zeros(self.max_hops + 1, dtype=np.float32)
            if n_layers == 1:
                result[0] = 1.0  # Self-transition probability
            return result

        # Build node-to-layer mapping
        node_to_layer: Dict[Any, int] = {}
        for hop, nodes in layers.items():
            for n in nodes:
                node_to_layer[n] = hop

        # Build transition probability matrix
        P = np.zeros((n_layers, n_layers), dtype=np.float32)

        for hop in range(n_layers):
            layer_nodes = layers.get(hop, [])

            edges_to_prev = 0
            edges_within = 0
            edges_to_next = 0

            for n in layer_nodes:
                for neighbor in G.neighbors(n):
                    neighbor_layer = node_to_layer.get(neighbor)
                    if neighbor_layer is None:
                        continue
                    if neighbor_layer == hop - 1:
                        edges_to_prev += 1
                    elif neighbor_layer == hop:
                        edges_within += 1
                    elif neighbor_layer == hop + 1:
                        edges_to_next += 1

            # Within-layer edges counted twice
            edges_within = edges_within // 2

            total = edges_to_prev + edges_within + edges_to_next

            if total > 0:
                if hop > 0:
                    P[hop, hop - 1] = edges_to_prev / total
                P[hop, hop] = edges_within / total
                if hop < n_layers - 1:
                    P[hop, hop + 1] = edges_to_next / total

        # Extract eigenvalues
        eigenvalues = np.linalg.eigvals(P)

        # Warn if eigenvalues have significant imaginary parts
        imag_parts = np.abs(np.imag(eigenvalues))
        if np.any(imag_parts > 1e-6):
            max_imag = np.max(imag_parts)
            warnings.warn(
                f"Eigenvalues have significant imaginary parts (max: {max_imag:.6f}). "
                "Only real parts will be used. This may indicate a non-symmetric "
                "transition matrix, which can occur with unusual graph structures.",
                UserWarning
            )

        eigenvalues = np.sort(np.real(eigenvalues))[::-1]  # Sort descending

        # Pad/truncate to fixed size
        result = np.zeros(self.max_hops + 1, dtype=np.float32)
        n_to_copy = min(len(eigenvalues), len(result))
        result[:n_to_copy] = eigenvalues[:n_to_copy]

        return result

    def compute(self, G: nx.Graph, verbose: bool = True) -> Dict[str, Any]:
        """
        Compute eigenvalue-based embeddings for all nodes.

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
            - "embeddings": dict mapping node_id to 1D eigenvalue vector
            - "method": "eigenvalue"
            - "params": algorithm parameters
            - "metadata": dict with layer info per node
        """
        if not isinstance(G, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")

        if G.is_directed():
            raise ValueError(
                "Eigenvalue method requires an undirected graph. "
                "The within-layer edge counting assumes each edge is seen from both endpoints. "
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
        return self._get_base_params()
