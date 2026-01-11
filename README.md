<p align="center">
  <h1 align="center">LSME</h1>
  <p align="center">
    <strong>Local Structural Matrix Embeddings</strong>
  </p>
  <p align="center">
    A Python library for generating interpretable structural embeddings of graph nodes
  </p>
</p>

<p align="center">
  <a href="https://github.com/elmspace/LSME/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9+-blue.svg"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg"></a>
  <a href="https://elmspace.github.io/LSME"><img alt="Documentation" src="https://img.shields.io/badge/docs-mkdocs-blue.svg"></a>
</p>

---

## Overview

**LSME** (Local Structural Matrix Embeddings) is a graph embedding technique that captures the local structural patterns around each node in a graph. Unlike traditional graph embedding methods that focus primarily on connectivity, LSME extracts rich structural information from node neighborhoods through interpretable *signature matrices*.

### Key Features

- **Interpretable Representations**: Signature matrices provide human-readable intermediate representations of local graph structure
- **Multiple Embedding Methods**: Four methods with different speed/quality trade-offs (stochastic, deterministic, random walk, eigenvalue)
- **Neural Encoders**: CNN and DNN autoencoders for encoding variable-sized neighborhoods into fixed-size embeddings
- **Flexible API**: Simple scikit-learn-inspired interface
- **Synthetic Graph Generation**: Built-in tools for creating benchmark graphs with known structural patterns

---

## What Problem Does LSME Solve?

Graph embeddings are essential for applying machine learning to graph-structured data. However, many existing methods:

1. **Lose structural information** by focusing only on node features or global connectivity
2. **Produce opaque embeddings** that are difficult to interpret
3. **Struggle with variable-sized neighborhoods** where different nodes have vastly different local structures

LSME addresses these challenges by:

1. Extracting **k-hop neighborhoods** around each node
2. Building **local adjacency matrices** organized by hop distance (layers)
3. Creating **signature matrices** through permutation averaging (stochastic method)
4. Encoding these matrices into **fixed-size embeddings** using neural autoencoders

The result is embeddings that capture meaningful structural patterns while remaining interpretable through the intermediate signature matrix representation.

---

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/elmspace/LSME.git
cd LSME
pip install -e .
```

### Using UV (Recommended)

```bash
git clone https://github.com/elmspace/LSME.git
cd LSME
uv pip install -e .
```

### With Development Dependencies

```bash
# For running tests
pip install -e ".[dev]"

# For building documentation
pip install -e ".[docs]"
```

### Requirements

- Python 3.9+
- NumPy >= 1.20.0
- NetworkX >= 2.5
- PyTorch >= 2.0.0

---

## Quick Start

```python
import networkx as nx
from lsme import LSME

# Load a graph
G = nx.karate_club_graph()

# Create LSME instance and compute embeddings
lsme = LSME(method='stochastic', max_hops=2, embedding_dim=32)
result = lsme.fit_transform(G)

# Access the embeddings
embeddings = result['embeddings']
print(f"Number of nodes: {len(embeddings)}")
print(f"Embedding dimension: {embeddings[0].shape}")  # (32,)

# For stochastic method, signature matrices are also available
signature_matrices = result['signature_matrices']
print(f"Signature matrix shape: {signature_matrices[0].shape}")
```

---

## Embedding Methods

LSME provides four embedding methods, each with different characteristics:

| Method | Description | Output Dimension | Speed | Quality |
|--------|-------------|------------------|-------|---------|
| `stochastic` | Signature matrices encoded via neural autoencoder | `embedding_dim` (configurable) | Slow | Highest |
| `deterministic` | Edge probability vectors across layers | `3 × (max_hops + 1)` | Very Fast | Medium |
| `random_walk` | Random walk transition probabilities | `3 × (max_hops + 1)` | Fast | Medium-High |
| `eigenvalue` | Spectral properties of transition matrix | `max_hops + 1` | Very Fast | Lower |

### Method Details

#### Stochastic (Default)

The flagship method that produces the highest quality embeddings:

```python
lsme = LSME(
    method='stochastic',
    max_hops=2,           # Neighborhood radius
    embedding_dim=32,     # Output embedding size
    n_samples=100,        # Permutation samples for averaging
    random_state=42       # For reproducibility
)
result = lsme.fit_transform(G)
```

**How it works:**
1. Extract k-hop BFS neighborhood organized into layers (layer 0 = root node, layer 1 = direct neighbors, etc.)
2. Generate local adjacency matrices with random within-layer permutations
3. Average across permutations to create a *signature matrix*
4. Encode signature matrices using a CNN/DNN autoencoder

#### Deterministic

Fast method requiring no training:

```python
lsme = LSME(method='deterministic', max_hops=2)
result = lsme.fit_transform(G)
# Output dimension: 9 (3 probabilities × 3 layers)
```

Computes edge probability vectors: probability of edges to previous layer, within current layer, and to next layer.

#### Random Walk

Stochastic method based on random walk transitions:

```python
lsme = LSME(
    method='random_walk',
    max_hops=2,
    rw_length=10,      # Steps per walk
    sample_size=100    # Number of walks
)
result = lsme.fit_transform(G)
```

#### Eigenvalue

Fastest method using spectral properties:

```python
lsme = LSME(method='eigenvalue', max_hops=2)
result = lsme.fit_transform(G)
# Output dimension: 3 (max_hops + 1)
```

---

## Advanced Usage

### Custom Encoder Configuration

For the stochastic method, you can configure the neural encoder:

```python
from lsme import LSME
from lsme.encoder import CNNEncoder, DNNEncoder

# Use CNN encoder (default) with custom settings
lsme = LSME(
    method='stochastic',
    embedding_dim=64,
    encoder_type='cnn',
    encoder_params={
        'hidden_channels': [32, 64, 128],
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'batch_size': 32
    }
)

# Or use DNN encoder
lsme = LSME(
    method='stochastic',
    encoder_type='dnn',
    encoder_params={
        'hidden_dims': [512, 256, 128]
    }
)
```

### Synthetic Graph Generation

LSME includes utilities for generating synthetic graphs with known structural patterns:

```python
from lsme.graphs import SyntheticGraphBuilder

# Build a composite graph with multiple patterns
builder = SyntheticGraphBuilder(random_state=42)
G = (builder
    .add_random(n_nodes=100, edge_prob=0.1)
    .add_barbell(count=5)
    .add_star_pattern(count=10)
    .add_web_pattern(count=3)
    .hydrate(prob=0.05)  # Add random inter-component edges
    .build())

# Nodes have 'role' and 'con_type' attributes for evaluation
roles = nx.get_node_attributes(G, 'role')
```

Available patterns: `barbell`, `star`, `web`, `dense_star`, `crossed_diamond`, `dynamic_star`

### Accessing Intermediate Results

The stochastic method provides rich intermediate outputs:

```python
result = lsme.fit_transform(G)

# Final embeddings
embeddings = result['embeddings']           # Dict[node_id → 1D array]

# Signature matrices (interpretable!)
sig_matrices = result['signature_matrices'] # Dict[node_id → 2D array]

# Layer information
layer_info = result['layer_info']           # Detailed layer metadata

# Trained encoder (can be saved/reused)
encoder = result['encoder']
encoder.save('my_encoder.pt')
```

---

## Examples

The `examples/` directory contains Jupyter notebooks demonstrating various use cases:

| Notebook | Description |
|----------|-------------|
| `01_quick_start.ipynb` | Basic usage and API introduction |
| `02_method_comparison.ipynb` | Compare all four embedding methods |
| `03_encoder_deep_dive.ipynb` | CNN vs DNN encoder analysis |
| `04_graph_patterns.ipynb` | Synthetic graph generation |
| `05_analysis_example.ipynb` | Node classification and clustering |
| `06_structural_roles.ipynb` | Identifying structural roles in graphs |
| `07_cross_graph_similarity.ipynb` | Cross-graph structural analysis |

To run the examples:

```bash
cd examples
pip install jupyterlab matplotlib plotly pandas scikit-learn umap-learn
jupyter lab
```

---

## Repository Structure

```
LSME/
├── src/                   # Python library
│   └── lsme/
│       ├── lsme.py        # Main LSME class
│       ├── core.py        # Core utilities (BFS, neighborhood extraction)
│       ├── methods/       # Embedding method implementations
│       │   ├── stochastic.py
│       │   ├── deterministic.py
│       │   ├── random_walk.py
│       │   └── eigenvalue.py
│       ├── encoder/       # Neural network encoders
│       │   ├── cnn_encoder.py
│       │   └── dnn_encoder.py
│       └── graphs/        # Synthetic graph generation
│           ├── builder.py
│           └── patterns.py
├── tests/                 # Pytest test suite
├── docs/                  # MkDocs documentation
├── examples/              # Jupyter notebook tutorials
└── pyproject.toml         # Package configuration
```

---

## Documentation

Full documentation is available at [elmspace.github.io/LSME](https://elmspace.github.io/LSME)

### Building Documentation Locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then visit `http://localhost:8000`

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=lsme --cov-report=term-missing

# Skip slow tests
pytest tests/ -m "not slow"
```

---

## Algorithm Overview

<p align="center">
  <img src="docs/assets/algorithm-flow.png" alt="LSME Algorithm Flow" width="700">
</p>

1. **Neighborhood Extraction**: For each node, extract its k-hop neighborhood using BFS, organizing nodes into layers by hop distance

2. **Local Adjacency Matrix**: Build a local adjacency matrix with the root node fixed at position 0, and other nodes ordered by layer

3. **Permutation Averaging** (Stochastic): Generate multiple random permutations within each layer and average the resulting matrices to create a *signature matrix* that captures structural patterns invariant to node ordering

4. **Encoding**: Use a CNN or DNN autoencoder to compress variable-sized signature matrices into fixed-size embedding vectors

---

## Citation

If you use LSME in your research, please cite:

```bibtex
@inproceedings{lsme2025,
  title={Local Structural Matrix Embeddings for Graph Representation Learning},
  author={},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## Acknowledgments

This work was supported by [acknowledgments to be added].
