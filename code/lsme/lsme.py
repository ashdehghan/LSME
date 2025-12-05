"""Main LSME class for generating structural embeddings."""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from typing import Optional, Union
from .core import compute_local_signature_matrix


class LSME:
    """
    Local Structural Matrix Embeddings for graphs.

    Generates structural embeddings for nodes in a graph by computing
    local signature matrices and optionally reducing their dimensionality.

    Parameters
    ----------
    max_hops : int, default=2
        Maximum hop distance to consider for local neighborhoods.

    n_samples : int, default=100
        Number of permutation samples to average for each signature matrix.

    embedding_dim : int, optional
        If specified, reduces signature matrices to this dimension using PCA.
        If None, returns flattened signature matrices.

    verbose : bool, default=True
        Whether to print progress information.

    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_hops: int = 2,
        n_samples: int = 100,
        embedding_dim: Optional[int] = None,
        verbose: bool = True,
        random_state: Optional[int] = None
    ):
        self.max_hops = max_hops
        self.n_samples = n_samples
        self.embedding_dim = embedding_dim
        self.verbose = verbose
        self.random_state = random_state
        self.pca = None

        if random_state is not None:
            np.random.seed(random_state)

    def fit_transform(self, G: nx.Graph) -> pd.DataFrame:
        """
        Generate embeddings for all nodes in the graph.

        Parameters
        ----------
        G : networkx.Graph
            The input graph.

        Returns
        -------
        pd.DataFrame
            DataFrame with node IDs as index and embedding dimensions as columns (e0, e1, ...).
        """
        if not isinstance(G, nx.Graph):
            raise ValueError("Input must be a NetworkX graph")

        if G.number_of_nodes() == 0:
            return pd.DataFrame()

        # Compute signature matrices for all nodes
        signatures = {}
        node_list = list(G.nodes())

        for i, node in enumerate(node_list):
            if self.verbose:
                print(f"Computing signature for node {node} ({i+1}/{len(node_list)})...")

            signatures[node] = compute_local_signature_matrix(
                G, node, self.max_hops, self.n_samples
            )

        # Flatten signature matrices
        flattened_signatures = {}
        max_size = 0

        for node, matrix in signatures.items():
            flat = matrix.flatten()
            flattened_signatures[node] = flat
            max_size = max(max_size, len(flat))

        # Pad to same size if needed
        for node in flattened_signatures:
            current = flattened_signatures[node]
            if len(current) < max_size:
                padded = np.zeros(max_size)
                padded[:len(current)] = current
                flattened_signatures[node] = padded

        # Create matrix of all embeddings
        embedding_matrix = np.array([flattened_signatures[node] for node in node_list])

        # Apply dimensionality reduction if requested
        if self.embedding_dim is not None and self.embedding_dim < embedding_matrix.shape[1]:
            if self.verbose:
                print(f"Reducing dimensions from {embedding_matrix.shape[1]} to {self.embedding_dim}...")

            self.pca = PCA(n_components=self.embedding_dim, random_state=self.random_state)
            embedding_matrix = self.pca.fit_transform(embedding_matrix)

        # Create DataFrame with proper column names
        n_dims = embedding_matrix.shape[1]
        columns = [f'e{i}' for i in range(n_dims)]

        df = pd.DataFrame(
            embedding_matrix,
            index=pd.Index(node_list, name='node_id'),
            columns=columns
        )

        if self.verbose:
            print(f"Generated embeddings with shape: {df.shape}")

        return df

    def transform(self, G: nx.Graph) -> pd.DataFrame:
        """
        Alias for fit_transform (no separate fitting needed for this algorithm).
        """
        return self.fit_transform(G)