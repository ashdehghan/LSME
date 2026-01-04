# API Reference

This section provides complete API documentation for the LSME library.

## Main Classes

| Class | Description |
|-------|-------------|
| [`LSME`](lsme.md) | Main class for computing node embeddings |
| [`CNNEncoder`](encoder/cnn_encoder.md) | CNN-based autoencoder for signature matrices |
| [`DNNEncoder`](encoder/dnn_encoder.md) | DNN-based autoencoder for signature matrices |
| [`SyntheticGraphBuilder`](graphs/builder.md) | Fluent builder for synthetic graphs |

## Quick Import

```python
from lsme import LSME, CNNEncoder, DNNEncoder, SyntheticGraphBuilder
```

## Module Structure

```
lsme/
├── __init__.py          # Main exports
├── core.py              # Core utilities
├── lsme.py              # LSME class
├── encoder/             # Encoder subpackage
│   ├── base.py          # BaseEncoder ABC
│   ├── cnn_encoder.py   # CNNEncoder
│   ├── dnn_encoder.py   # DNNEncoder
│   ├── model.py         # PyTorch models
│   └── dataset.py       # Dataset utilities
├── methods/             # Embedding methods
│   ├── base.py          # BaseMethod ABC
│   ├── stochastic.py    # Stochastic method
│   ├── deterministic.py # Deterministic method
│   ├── random_walk.py   # Random walk method
│   └── eigenvalue.py    # Eigenvalue method
└── graphs/              # Graph utilities
    ├── builder.py       # SyntheticGraphBuilder
    └── patterns.py      # Pattern functions
```
