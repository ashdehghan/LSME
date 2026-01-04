# Quick Start

This guide will get you up and running with LSME in 5 minutes.

## Basic Usage

```python
import networkx as nx
from lsme import LSME

# Load a graph (Zachary's Karate Club)
G = nx.karate_club_graph()

# Create LSME embedder with default settings
lsme = LSME(max_hops=2, embedding_dim=32)

# Compute embeddings
result = lsme.fit_transform(G)

# Access the embeddings
embeddings = result['embeddings']
print(f"Number of nodes: {len(embeddings)}")
print(f"Embedding shape: {embeddings[0].shape}")
```

## Understanding the Output

The `fit_transform()` method returns a dictionary with:

```python
result = lsme.fit_transform(G)

# Always present
result['embeddings']      # Dict: node_id -> 1D numpy array
result['method']          # String: method name ('stochastic', etc.)
result['params']          # Dict: algorithm parameters used
result['metadata']        # Dict: method-specific metadata

# Stochastic method only
result['signature_matrices']  # Dict: node_id -> 2D signature matrix
result['layer_info']          # Dict: node_id -> layer structure info
result['encoder']             # Trained encoder instance
```

## Choosing a Method

LSME supports four embedding methods:

=== "Stochastic (Default)"

    Best for capturing probabilistic local structure:

    ```python
    lsme = LSME(
        method='stochastic',
        max_hops=2,
        n_samples=100,      # Number of permutations to average
        embedding_dim=32    # Output embedding dimension
    )
    ```

=== "Deterministic"

    Faster, based on transition probabilities:

    ```python
    lsme = LSME(
        method='deterministic',
        max_hops=3
    )
    # Output dimension: 3 * (max_hops + 1)
    ```

=== "Random Walk"

    Stochastic sampling of transitions:

    ```python
    lsme = LSME(
        method='random_walk',
        max_hops=2,
        rw_length=10,      # Random walk length
        sample_size=100    # Number of walks
    )
    ```

=== "Eigenvalue"

    Spectral approach:

    ```python
    lsme = LSME(
        method='eigenvalue',
        max_hops=3
    )
    # Output dimension: max_hops + 1
    ```

## Visualizing Embeddings

Use dimensionality reduction to visualize embeddings:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Get embeddings as matrix
nodes = sorted(result['embeddings'].keys())
X = np.array([result['embeddings'][n] for n in nodes])

# Reduce to 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1])
for i, node in enumerate(nodes):
    plt.annotate(str(node), (X_2d[i, 0], X_2d[i, 1]))
plt.title("LSME Embeddings (PCA)")
plt.show()
```

## Reproducibility

Use `random_state` for reproducible results:

```python
lsme = LSME(
    max_hops=2,
    n_samples=100,
    random_state=42
)
```

## Next Steps

- [Learn about the algorithm](../concepts/algorithm.md)
- [Compare embedding methods](../concepts/methods.md)
- [Explore the API](../api/index.md)
- [Try the example notebooks](../examples/index.md)
