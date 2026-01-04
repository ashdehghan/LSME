# LSME - Local Structural Matrix Embeddings

A Python library for generating structural embeddings of nodes in graphs.

For full documentation, see the [main README](../README.md) or build the docs:

```bash
pip install -e ".[docs]"
mkdocs serve  # from repo root
```

## Installation

```bash
pip install -e .

# With dev dependencies (pytest)
pip install -e ".[dev]"

# With docs dependencies (mkdocs)
pip install -e ".[docs]"
```

## Quick Start

```python
import networkx as nx
from lsme import LSME

# Load a graph
G = nx.karate_club_graph()

# Compute embeddings (default: stochastic method)
lsme = LSME(max_hops=2, embedding_dim=32)
result = lsme.fit_transform(G)

# Access results
embeddings = result['embeddings']           # Dict: node -> 1D array
signature_matrices = result['signature_matrices']  # Dict: node -> 2D array
```

## Available Methods

```python
# Stochastic (default) - highest quality
lsme = LSME(method='stochastic', max_hops=2, n_samples=100, embedding_dim=32)

# Deterministic - fast, no training
lsme = LSME(method='deterministic', max_hops=3)

# Random walk - stochastic, no training
lsme = LSME(method='random_walk', max_hops=2, rw_length=10)

# Eigenvalue - spectral approach
lsme = LSME(method='eigenvalue', max_hops=3)
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Embedding method | `'stochastic'` |
| `max_hops` | Neighborhood radius | `2` |
| `n_samples` | Permutation samples (stochastic) | `100` |
| `embedding_dim` | Output dimension (stochastic) | `32` |
| `encoder_type` | `'cnn'` or `'dnn'` | `'cnn'` |
| `random_state` | Reproducibility seed | `None` |

## Package Structure

```
lsme/
├── __init__.py      # Public API exports
├── lsme.py          # Main LSME class
├── core.py          # Core utilities
├── encoder/         # CNN/DNN autoencoders
├── methods/         # Embedding methods
└── graphs/          # Synthetic graph generation
```
