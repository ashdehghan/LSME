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
cd code && python test_encoder.py
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

The LSME library generates structural signature matrices for graph nodes:

1. **`core.py`** - Low-level algorithm functions:
   - `get_nodes_by_hop_distance()` - BFS to organize nodes by hop distance from root
   - `build_local_adjacency_matrix()` - Constructs adjacency matrix with layer-based ordering
   - `compute_local_signature_matrix()` - Averages multiple permuted local adjacency matrices, returns (matrix, layers) tuple

2. **`lsme.py`** - Main `LSME` class providing the public API:
   - `fit_transform(G)` - Computes signature matrices for all nodes
   - Returns dict with: `signature_matrices` (node → 2D array), `layer_info` (node → layer metadata), `params`

3. **`encoder/`** - CNN autoencoder subpackage for encoding signature matrices:
   - `SignatureEncoder` - Main class for training autoencoder and extracting embeddings
   - `SignatureAutoencoder` - PyTorch CNN model (encoder + decoder)
   - `SignatureDataset` - PyTorch Dataset with padding and masking
   - Uses masked MSE loss to handle variable-sized matrices

The algorithm: for each node, extract k-hop neighborhood, build local adjacency matrices with random within-layer permutations, average these matrices to produce a signature matrix. Optionally, use the CNN autoencoder to encode matrices into fixed-size embeddings.

## Dependencies

- numpy, networkx, torch
- Uses hatchling for builds

## Experimentation

The `experimentation/test/` directory contains validation experiments (Jupyter notebooks). To run:

```bash
cd experimentation/test
uv venv && uv pip install -e ../../code && uv pip install .
uv run jupyter lab
```
