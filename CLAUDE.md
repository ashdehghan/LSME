# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

```
LSME/
├── code/              # Python library for LSME
│   └── lsme/          # Core package
├── tests/             # Pytest test suite
├── docs/              # MkDocs documentation
├── examples/          # Jupyter notebook examples
├── manuscript/        # LaTeX paper (ICML 2025)
└── experimentation/   # Research notebooks
```

## Common Commands

### Python Code

```bash
# Install in development mode
cd code && pip install -e .

# Install with dev dependencies (pytest)
cd code && pip install -e ".[dev]"

# Install with docs dependencies (mkdocs)
cd code && pip install -e ".[docs]"
```

### Running Tests

```bash
# Run all tests (from repo root)
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_lsme.py

# Run with coverage
pytest tests/ --cov=lsme --cov-report=term-missing

# Skip slow tests
pytest tests/ -m "not slow"
```

### Documentation

```bash
# Install docs dependencies
pip install -e "./code[docs]"

# Serve locally (from repo root)
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

### Examples

```bash
# Navigate to examples
cd examples

# Install dependencies
pip install -e ../code
pip install jupyterlab matplotlib plotly pandas scikit-learn umap-learn

# Start Jupyter
jupyter lab
```

### Manuscript

```bash
cd manuscript
make        # Full build (pdflatex + bibtex)
make quick  # Quick compile without bibtex
make clean  # Remove build artifacts
make view   # Open PDF (macOS)
```

## Code Architecture

The LSME library generates structural embeddings for graph nodes:

### Main Components

1. **`lsme.py`** - Main `LSME` class with unified API
   - `fit_transform(G)` - Computes embeddings for all nodes
   - Supports 4 methods: stochastic, deterministic, random_walk, eigenvalue

2. **`core.py`** - Low-level utilities
   - `get_nodes_by_hop_distance()` - BFS for neighborhood extraction

3. **`methods/`** - Embedding method implementations
   - `stochastic.py` - Signature matrix averaging
   - `deterministic.py` - Edge probability vectors
   - `random_walk.py` - Random walk transitions
   - `eigenvalue.py` - Spectral properties

4. **`encoder/`** - Neural network encoders
   - `cnn_encoder.py` - CNN autoencoder
   - `dnn_encoder.py` - DNN autoencoder
   - Handles variable-sized matrices via padding/masking

5. **`graphs/`** - Synthetic graph generation
   - `builder.py` - Fluent SyntheticGraphBuilder
   - `patterns.py` - Pattern functions (barbell, star, web, etc.)

### Algorithm Summary

1. Extract k-hop neighborhood for each node
2. Build local adjacency matrices with layer-based ordering
3. (Stochastic) Average across random within-layer permutations
4. (Stochastic) Encode signature matrices to fixed-size embeddings

## Dependencies

- numpy, networkx, torch
- pytest (dev), mkdocs (docs)
- Uses hatchling for builds
