# Random Walk Method

The random walk method uses random walks to estimate transition probabilities between layers.

## Usage

```python
from lsme import LSME

lsme = LSME(
    method='random_walk',
    max_hops=2,
    rw_length=10,
    sample_size=100
)
result = lsme.fit_transform(G)
```

## API Reference

::: lsme.methods.RandomWalkMethod
    options:
      show_root_heading: true
      show_source: true
