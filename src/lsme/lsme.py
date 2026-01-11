"""Main LSME class for computing local structural embeddings."""

import random
import numpy as np
import networkx as nx
from typing import Any, Dict, Optional, Type

from .methods.base import BaseMethod
from .methods.stochastic import StochasticMethod
from .methods.deterministic import DeterministicMethod
from .methods.random_walk import RandomWalkMethod
from .methods.eigenvalue import EigenvalueMethod
from .encoder import CNNEncoder, DNNEncoder, BaseEncoder


# Registry of available embedding methods
METHODS: Dict[str, Type[BaseMethod]] = {
    'stochastic': StochasticMethod,
    'deterministic': DeterministicMethod,
    'random_walk': RandomWalkMethod,
    'eigenvalue': EigenvalueMethod,
}

# Registry of available encoders (for stochastic method)
ENCODERS: Dict[str, Type[BaseEncoder]] = {
    'cnn': CNNEncoder,
    'dnn': DNNEncoder,
}


class LSME:
    """
    Local Structural Matrix Embeddings for graphs.

    Computes local structural embeddings for nodes in a graph using
    various embedding methods.

    Parameters
    ----------
    method : str, default='stochastic'
        The embedding method to use. Available options:
        - 'stochastic': Averaged permuted adjacency matrices encoded via autoencoder
        - 'deterministic': Transition probability vectors based on edge counts
        - 'random_walk': Transition probabilities from random walks
        - 'eigenvalue': Eigenvalues of layer transition matrices

    max_hops : int, default=2
        Maximum hop distance to consider for local neighborhoods.

    n_samples : int, default=100
        Number of permutation samples (for 'stochastic' method).

    rw_length : int, default=10
        Length of random walks (for 'random_walk' method).

    sample_size : int, default=100
        Number of random walks per node (for 'random_walk' method).

    encoder_type : str, default='cnn'
        Type of encoder for stochastic method ('cnn' or 'dnn').

    embedding_dim : int, default=32
        Dimension of output embeddings (for 'stochastic' method).

    encoder_epochs : int, default=100
        Number of training epochs for the encoder (for 'stochastic' method).

    encoder_kwargs : dict, optional
        Additional keyword arguments passed to the encoder (e.g., learning_rate,
        batch_size, hidden_channels for CNN, hidden_dims for DNN).

    verbose : bool, default=True
        Whether to print progress information.

    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> import networkx as nx
    >>> from lsme import LSME
    >>> G = nx.karate_club_graph()
    >>> # Using stochastic method (default) - returns encoded embeddings
    >>> lsme = LSME(method='stochastic', max_hops=2, embedding_dim=32)
    >>> result = lsme.fit_transform(G)
    >>> result['embeddings'][0].shape  # 1D encoded embedding
    (32,)
    >>> result['signature_matrices'][0].shape  # Raw 2D signature matrix
    (15, 15)
    >>> # Using deterministic method
    >>> lsme = LSME(method='deterministic', max_hops=3)
    >>> result = lsme.fit_transform(G)
    >>> result['embeddings'][0].shape  # 1D probability vector
    (12,)
    """

    def __init__(
        self,
        method: str = 'stochastic',
        max_hops: int = 2,
        n_samples: int = 100,
        rw_length: int = 10,
        sample_size: int = 100,
        encoder_type: str = 'cnn',
        embedding_dim: int = 32,
        encoder_epochs: int = 100,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        random_state: Optional[int] = None
    ):
        if method not in METHODS:
            valid_methods = ', '.join(sorted(METHODS.keys()))
            raise ValueError(
                f"Unknown method: '{method}'. Valid methods are: {valid_methods}"
            )

        if encoder_type not in ENCODERS:
            valid_encoders = ', '.join(sorted(ENCODERS.keys()))
            raise ValueError(
                f"Unknown encoder_type: '{encoder_type}'. Valid encoders are: {valid_encoders}"
            )

        self.method = method
        self.max_hops = max_hops
        self.n_samples = n_samples
        self.rw_length = rw_length
        self.sample_size = sample_size
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        self.encoder_epochs = encoder_epochs
        self.encoder_kwargs = encoder_kwargs or {}
        self.verbose = verbose
        self.random_state = random_state

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def _create_method_instance(self) -> BaseMethod:
        """Create an instance of the selected embedding method."""
        method_cls = METHODS[self.method]

        # Build kwargs based on method type
        kwargs: Dict[str, Any] = {
            'max_hops': self.max_hops,
            'random_state': self.random_state,
        }

        if self.method == 'stochastic':
            kwargs['n_samples'] = self.n_samples
        elif self.method == 'random_walk':
            kwargs['rw_length'] = self.rw_length
            kwargs['sample_size'] = self.sample_size

        return method_cls(**kwargs)

    def _create_encoder(self) -> BaseEncoder:
        """Create an encoder instance for the stochastic method."""
        encoder_cls = ENCODERS[self.encoder_type]

        kwargs = {
            'embedding_dim': self.embedding_dim,
            'num_epochs': self.encoder_epochs,
            'verbose': self.verbose,
            'random_state': self.random_state,
        }
        kwargs.update(self.encoder_kwargs)

        return encoder_cls(**kwargs)

    def fit_transform(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Compute embeddings for all nodes in the graph.

        Parameters
        ----------
        G : networkx.Graph
            The input graph.

        Returns
        -------
        dict
            Dictionary containing:
            - "embeddings": dict mapping node_id to embedding (1D array)
            - "method": str, the method name used
            - "params": dict with algorithm parameters
            - "metadata": dict mapping node_id to method-specific metadata

            For 'stochastic' method, also includes:
            - "signature_matrices": raw 2D signature matrices
            - "layer_info": detailed layer information per node
            - "encoder": trained encoder instance
        """
        if not isinstance(G, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")

        # Stochastic method requires minimum 3 nodes for meaningful encoder training
        # Empty graphs (0 nodes) are handled gracefully by returning empty results
        n_nodes = G.number_of_nodes()
        if self.method == 'stochastic' and 0 < n_nodes < 3:
            raise ValueError(
                f"Stochastic method requires at least 3 nodes for encoder training. "
                f"Graph has {n_nodes} node(s). "
                f"Use 'deterministic', 'random_walk', or 'eigenvalue' method for small graphs."
            )

        method_instance = self._create_method_instance()
        result = method_instance.compute(G, verbose=self.verbose)

        # For stochastic method, encode signature matrices to get fixed-size embeddings
        if self.method == 'stochastic' and G.number_of_nodes() > 0:
            if self.verbose:
                print("\nTraining encoder...")

            encoder = self._create_encoder()
            embeddings = encoder.fit_transform(
                result['signature_matrices'],
                result['layer_info']
            )
            result['embeddings'] = embeddings
            result['encoder'] = encoder

        return result

    def transform(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Alias for fit_transform (no separate fitting needed for this algorithm).

        Parameters
        ----------
        G : networkx.Graph
            The input graph.

        Returns
        -------
        dict
            Same as fit_transform.
        """
        return self.fit_transform(G)

    @staticmethod
    def available_methods() -> list:
        """
        Return list of available embedding methods.

        Returns
        -------
        list
            List of method names.
        """
        return sorted(METHODS.keys())
