# Methods Overview

The methods subpackage provides different algorithms for computing node embeddings.

## Available Methods

| Method | Description | Output Dim |
|--------|-------------|------------|
| [`Stochastic`](stochastic.md) | Permutation-averaged signature matrices | `embedding_dim` |
| [`Deterministic`](deterministic.md) | Edge probability vectors | `3*(max_hops+1)` |
| [`Random Walk`](random_walk.md) | Random walk transition probabilities | `3*(max_hops+1)` |
| [`Eigenvalue`](eigenvalue.md) | Transition matrix eigenvalues | `max_hops+1` |

## Selecting a Method

```python
from lsme import LSME

# Specify method at construction
lsme = LSME(method='stochastic')  # default
lsme = LSME(method='deterministic')
lsme = LSME(method='random_walk')
lsme = LSME(method='eigenvalue')

# List available methods
print(LSME.available_methods())
# ['stochastic', 'deterministic', 'random_walk', 'eigenvalue']
```
