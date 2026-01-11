"""Abstract base class for LSME embedding methods."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import networkx as nx
import numpy as np


class BaseMethod(ABC):
    """
    Abstract base class for LSME embedding methods.

    All embedding methods must inherit from this class and implement
    the required abstract methods.

    Parameters
    ----------
    max_hops : int, default=2
        Maximum hop distance to consider for local neighborhoods.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, max_hops: int = 2, random_state: Optional[int] = None):
        self.max_hops = max_hops
        self.random_state = random_state

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the method identifier string."""
        pass

    @abstractmethod
    def compute(self, G: nx.Graph, verbose: bool = True) -> Dict[str, Any]:
        """
        Compute embeddings for all nodes in the graph.

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
            - "embeddings": dict mapping node_id to numpy array
            - "method": str, the method name
            - "params": dict with algorithm parameters
            - "metadata": dict mapping node_id to method-specific metadata
            Plus any method-specific additional keys.
        """
        pass

    @abstractmethod
    def compute_node(self, G: nx.Graph, node: Any) -> np.ndarray:
        """
        Compute embedding for a single node.

        Parameters
        ----------
        G : networkx.Graph
            The input graph.
        node : Any
            The node to compute embedding for.

        Returns
        -------
        np.ndarray
            The embedding vector or matrix for the node.
        """
        pass

    def _get_base_params(self) -> Dict[str, Any]:
        """Return base parameters common to all methods."""
        return {
            "max_hops": self.max_hops,
            "random_state": self.random_state,
        }
