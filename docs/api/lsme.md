# LSME Class

The main class for computing local structural matrix embeddings.

## Usage

```python
from lsme import LSME
import networkx as nx

G = nx.karate_club_graph()

# Default: stochastic method
lsme = LSME(max_hops=2, embedding_dim=32)
result = lsme.fit_transform(G)

# Other methods
lsme_det = LSME(method='deterministic', max_hops=3)
lsme_rw = LSME(method='random_walk', max_hops=2, rw_length=10)
lsme_eig = LSME(method='eigenvalue', max_hops=3)
```

## API Reference

::: lsme.LSME
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - fit_transform
        - transform
        - available_methods
