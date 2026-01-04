# LSME: Local Structural Matrix Embeddings

LSME is a Python library for generating structural embeddings of nodes in graphs. It captures local neighborhood patterns around each node and converts them into fixed-size vector representations suitable for machine learning tasks.

## Features

- **Multiple embedding methods**: Stochastic (signature matrices), deterministic, random walk, and eigenvalue-based approaches
- **Flexible encoders**: CNN and DNN autoencoders for encoding variable-sized signature matrices
- **Synthetic graph generation**: Built-in tools for creating test graphs with known structural patterns
- **Clean API**: Simple, scikit-learn-inspired interface

## Quick Example

```python
import networkx as nx
from lsme import LSME

# Load a graph
G = nx.karate_club_graph()

# Compute embeddings
lsme = LSME(method='stochastic', max_hops=2, embedding_dim=32)
result = lsme.fit_transform(G)

# Access embeddings
embeddings = result['embeddings']
print(f"Node 0 embedding shape: {embeddings[0].shape}")  # (32,)
```

## Installation

```bash
pip install lsme
```

Or install from source:

```bash
git clone https://github.com/yourusername/LSME.git
cd LSME/code
pip install -e .
```

## When to Use LSME

LSME is particularly useful when:

- You need to capture **local structural patterns** around nodes
- Your graph has nodes with **similar local neighborhoods** that should cluster together
- You want **interpretable intermediate representations** (signature matrices)
- You need **fixed-size embeddings** for variable-sized neighborhoods

## Getting Started

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quick Start__

    ---

    Get up and running with LSME in 5 minutes

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } __Concepts__

    ---

    Understand how LSME works and when to use each method

    [:octicons-arrow-right-24: Learn More](concepts/algorithm.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Complete API documentation with examples

    [:octicons-arrow-right-24: API Docs](api/index.md)

-   :material-notebook:{ .lg .middle } __Examples__

    ---

    Jupyter notebooks with detailed walkthroughs

    [:octicons-arrow-right-24: Examples](examples/index.md)

</div>
