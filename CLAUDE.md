# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This is a research project with two main components:
- `code/` - Python library for Local Structural Matrix Embeddings
- `manuscript/` - LaTeX paper (ICML 2025 format)

## Common Commands

### Python Code (in `code/` directory)

```bash
# Install in development mode
cd code && pip install -e .

# Run tests
cd code && python test_lsme.py
```

### Manuscript (in `manuscript/` directory)

```bash
cd manuscript
make        # Full build (pdflatex + bibtex)
make quick  # Quick compile without bibtex
make clean  # Remove build artifacts
make view   # Open PDF (macOS)
```

## Code Architecture

The LSME library generates structural embeddings for graph nodes:

1. **`core.py`** - Low-level algorithm functions:
   - `get_nodes_by_hop_distance()` - BFS to organize nodes by hop distance from root
   - `build_local_adjacency_matrix()` - Constructs adjacency matrix with layer-based ordering
   - `compute_local_signature_matrix()` - Averages multiple permuted local adjacency matrices

2. **`lsme.py`** - Main `LSME` class providing the public API:
   - `fit_transform(G)` - Computes embeddings for all nodes, returns pandas DataFrame
   - Handles flattening signature matrices, padding to uniform size, optional PCA reduction

The algorithm works by: for each node, extracting its k-hop neighborhood, building local adjacency matrices with random within-layer permutations, averaging these matrices, then flattening into embeddings.

## Dependencies

- numpy, networkx, pandas, scikit-learn
- Uses hatchling for builds
