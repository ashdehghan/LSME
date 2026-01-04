# Signature Matrices

Signature matrices are the intermediate representation in LSME that capture local structural patterns around each node.

## What is a Signature Matrix?

A signature matrix is a 2D array where:

- Rows and columns correspond to positions in the node's local neighborhood
- Values represent edge probabilities between those positions
- The matrix is symmetric (for undirected graphs)

## Structure

Given a node with `max_hops=2`, the signature matrix is organized as:

```
         Layer 0  |  Layer 1  |  Layer 2
        ┌─────────┼───────────┼──────────┐
Layer 0 │  0      │  edges    │  0       │
        ├─────────┼───────────┼──────────┤
Layer 1 │  edges  │  within   │  edges   │
        ├─────────┼───────────┼──────────┤
Layer 2 │  0      │  edges    │  within  │
        └─────────┴───────────┴──────────┘
```

- **Layer 0**: Always just the root node (self-loop = 0)
- **Layer 1**: Direct neighbors of root
- **Layer 2**: Nodes at distance 2

## Interpreting Values

After averaging across permutations:

- **Value = 1.0**: Edge always present between these positions
- **Value = 0.5**: Edge present 50% of the time (depends on permutation)
- **Value = 0.0**: Edge never present

## Example: Hub vs Bridge Node

### Hub Node (high degree)

```python
# Node 0 in a star graph
#       1
#       |
#   4 - 0 - 2
#       |
#       3

# Signature matrix (simplified):
[[0.0, 1.0, 1.0, 1.0, 1.0],   # Root connected to all
 [1.0, 0.0, 0.0, 0.0, 0.0],   # Leaves not connected
 [1.0, 0.0, 0.0, 0.0, 0.0],   # to each other
 [1.0, 0.0, 0.0, 0.0, 0.0],
 [1.0, 0.0, 0.0, 0.0, 0.0]]
```

### Bridge Node (in a chain)

```python
# Node 1 in a path: 0 - 1 - 2 - 3
# With max_hops=2:
# Layer 0: {1}
# Layer 1: {0, 2}
# Layer 2: {3}

# Signature matrix:
[[0.0, 1.0, 1.0, 0.0],   # Root to layer 1
 [1.0, 0.0, 0.0, 0.0],   # Layer 1 nodes
 [1.0, 0.0, 0.0, 1.0],   # Node 2 connects to layer 2
 [0.0, 0.0, 1.0, 0.0]]   # Layer 2
```

## Accessing Signature Matrices

```python
from lsme import LSME
import networkx as nx

G = nx.karate_club_graph()
lsme = LSME(method='stochastic', max_hops=2, n_samples=100)
result = lsme.fit_transform(G)

# Get signature matrix for node 0
sig_matrix = result['signature_matrices'][0]
print(f"Shape: {sig_matrix.shape}")

# Get layer information
layer_info = result['layer_info'][0]
print(f"Layers: {layer_info['layers']}")
print(f"Total nodes: {layer_info['total_nodes']}")
```

## Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for idx, node in enumerate([0, 1, 33]):
    ax = axes[idx]
    im = ax.imshow(result['signature_matrices'][node], cmap='Blues')
    ax.set_title(f'Node {node}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

## Why Signature Matrices Matter

1. **Interpretability**: You can visualize and understand what the model sees
2. **Debugging**: Check if neighborhoods are being captured correctly
3. **Analysis**: Compare structural patterns across nodes
4. **Transfer**: Train encoder on one graph, apply to another

## Variable Sizes

Signature matrices vary in size because neighborhoods vary:

- **Large neighborhood**: Large matrix
- **Small neighborhood**: Small matrix
- **Isolated node**: 1x1 matrix (just the root)

The encoder handles this by padding and masking during training.
