# Examples

This section provides hands-on examples and tutorials for using LSME.

## Jupyter Notebooks

The `examples/` directory contains comprehensive Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| [01_quick_start.ipynb](notebooks.md#quick-start) | Basic usage with Karate Club graph |
| [02_method_comparison.ipynb](notebooks.md#method-comparison) | Compare all 4 embedding methods |
| [03_encoder_deep_dive.ipynb](notebooks.md#encoder-deep-dive) | CNN vs DNN encoders, hyperparameter tuning |
| [04_graph_patterns.ipynb](notebooks.md#graph-patterns) | SyntheticGraphBuilder usage |
| [05_analysis_example.ipynb](notebooks.md#analysis-example) | Classification and clustering with embeddings |

## Quick Examples

### Basic Usage

```python
import networkx as nx
from lsme import LSME

G = nx.karate_club_graph()
lsme = LSME(max_hops=2, embedding_dim=32)
result = lsme.fit_transform(G)

# Access embeddings
for node, emb in result['embeddings'].items():
    print(f"Node {node}: {emb.shape}")
```

### Visualize Signature Matrices

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for idx, node in enumerate([0, 1, 33]):
    im = axes[idx].imshow(result['signature_matrices'][node], cmap='Blues')
    axes[idx].set_title(f'Node {node}')
    plt.colorbar(im, ax=axes[idx])
plt.tight_layout()
plt.show()
```

### 2D Projection

```python
from sklearn.decomposition import PCA
import numpy as np

nodes = sorted(result['embeddings'].keys())
X = np.array([result['embeddings'][n] for n in nodes])

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=range(len(nodes)), cmap='viridis')
plt.colorbar(label='Node ID')
plt.title('LSME Embeddings (PCA)')
plt.show()
```

### Synthetic Graphs

```python
from lsme import SyntheticGraphBuilder

G = (SyntheticGraphBuilder(random_state=42)
     .add_barbell(count=3)
     .add_star_pattern(count=5)
     .hydrate(prob=0.05)
     .build())

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
```

### Node Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Prepare data
X = np.array([result['embeddings'][n] for n in sorted(G.nodes())])
y = np.array([G.nodes[n]['role'] for n in sorted(G.nodes())])

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```
