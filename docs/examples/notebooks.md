# Jupyter Notebooks

Detailed Jupyter notebooks are available in the `examples/` directory.

## Running the Notebooks

```bash
# Navigate to examples directory
cd examples

# Install dependencies
pip install -e ../code
pip install matplotlib plotly pandas scikit-learn umap-learn jupyterlab

# Start Jupyter
jupyter lab
```

## Available Notebooks

### Quick Start

**File:** `01_quick_start.ipynb`

A beginner-friendly introduction to LSME:

- Loading graphs (Karate Club, custom graphs)
- Computing embeddings with default settings
- Visualizing the graph and signature matrices
- 2D projection of embeddings
- Using different methods

### Method Comparison

**File:** `02_method_comparison.ipynb`

Compare all 4 embedding methods:

- Stochastic, deterministic, random walk, eigenvalue
- Side-by-side 2D projections
- Quantitative comparison (silhouette scores)
- When to use each method

### Encoder Deep Dive

**File:** `03_encoder_deep_dive.ipynb`

Detailed exploration of the encoding step:

- CNN vs DNN architecture
- Training curves and convergence
- Reconstruction quality visualization
- Hyperparameter tuning (embedding_dim, epochs)
- Saving and loading models

### Graph Patterns

**File:** `04_graph_patterns.ipynb`

Using SyntheticGraphBuilder:

- Individual pattern demos (barbell, web, star, etc.)
- Composing complex graphs
- Hydration and trimming
- Verifying LSME captures structural roles

### Analysis Example

**File:** `05_analysis_example.ipynb`

End-to-end machine learning example:

- Creating labeled synthetic graphs
- Computing embeddings
- Unsupervised clustering (K-Means, hierarchical)
- Supervised classification (Random Forest, SVM, KNN)
- Comparing methods for downstream tasks
