# LSME: Local Structural Matrix Embeddings

A Python library for generating structural embeddings of nodes in graphs.

## Features

- **Multiple embedding methods**: Stochastic, deterministic, random walk, eigenvalue
- **Flexible encoders**: CNN and DNN autoencoders for encoding signature matrices
- **Synthetic graph generation**: Built-in tools for creating test graphs
- **Clean API**: Simple, scikit-learn-inspired interface

## Quick Start

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
print(f"Node 0 embedding: {embeddings[0].shape}")  # (32,)
```

## Installation

```bash
# Install from source
git clone https://github.com/elmspace/LSME.git
cd LSME/code
pip install -e .
```

## Repository Structure

```
LSME/
├── code/              # Python library
│   └── lsme/          # Core package
├── tests/             # Pytest test suite
├── docs/              # MkDocs documentation
├── examples/          # Jupyter notebook examples
├── manuscript/        # LaTeX paper (ICML 2025)
└── experimentation/   # Research notebooks
```

## Documentation

- **[Quick Start](docs/getting-started/quickstart.md)** - Get up and running
- **[API Reference](docs/api/index.md)** - Complete API documentation
- **[Concepts](docs/concepts/algorithm.md)** - Understand the algorithm

Build docs locally:

```bash
pip install -e "./code[docs]"
mkdocs serve
```

## Examples

See the `examples/` directory for Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| `01_quick_start.ipynb` | Basic usage tutorial |
| `02_method_comparison.ipynb` | Compare embedding methods |
| `03_encoder_deep_dive.ipynb` | CNN vs DNN encoders |
| `04_graph_patterns.ipynb` | Synthetic graph generation |
| `05_analysis_example.ipynb` | Classification and clustering |

## Testing

```bash
pip install -e "./code[dev]"
pytest tests/
```

## Methods

| Method | Description | Output Dim |
|--------|-------------|------------|
| `stochastic` | Signature matrices + encoder | `embedding_dim` |
| `deterministic` | Edge probability vectors | `3*(max_hops+1)` |
| `random_walk` | Random walk transitions | `3*(max_hops+1)` |
| `eigenvalue` | Spectral properties | `max_hops+1` |

## Citation

TBD

## License

TBD
