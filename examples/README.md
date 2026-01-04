# LSME Examples

This directory contains Jupyter notebooks demonstrating LSME features.

## Setup

```bash
# Install LSME library
pip install -e ../code

# Install example dependencies
pip install matplotlib plotly pandas scikit-learn umap-learn seaborn jupyterlab

# Or use the pyproject.toml
pip install -e .

# Start Jupyter
jupyter lab
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_quick_start.ipynb` | Basic usage with Karate Club graph |
| `02_method_comparison.ipynb` | Compare all 4 embedding methods |
| `03_encoder_deep_dive.ipynb` | CNN vs DNN encoders, hyperparameter tuning |
| `04_graph_patterns.ipynb` | SyntheticGraphBuilder usage |
| `05_analysis_example.ipynb` | Classification and clustering with embeddings |

## Running All Notebooks

To verify all notebooks run without errors:

```bash
jupyter nbconvert --execute --to notebook *.ipynb
```
