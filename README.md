# LSME - Local Structural Matrix Embeddings

A simple and efficient library for generating structural embeddings of nodes in graphs.

## Installation

```bash
pip install -e .
```

## Usage

```python
import networkx as nx
from lsme import LSME

# Create or load your graph
G = nx.karate_club_graph()

# Initialize LSME
embedder = LSME(
    max_hops=2,        # Neighborhood depth
    n_samples=100,     # Number of permutations to average
    embedding_dim=16,  # Optional: reduce to this dimension
    verbose=True       # Print progress
)

# Generate embeddings
embeddings = embedder.fit_transform(G)

# Result is a pandas DataFrame with node_id as index
# and embedding dimensions (e0, e1, ...) as columns
print(embeddings.head())
```

## Parameters

- `max_hops`: Maximum hop distance for local neighborhoods (default: 2)
- `n_samples`: Number of permutation samples to average (default: 100)
- `embedding_dim`: Optional dimensionality reduction via PCA
- `verbose`: Print progress information (default: True)
- `random_state`: Random seed for reproducibility

## Output

Returns a pandas DataFrame with:
- Index: `node_id` (node identifiers from the graph)
- Columns: `e0`, `e1`, ..., `ek` (embedding dimensions)
