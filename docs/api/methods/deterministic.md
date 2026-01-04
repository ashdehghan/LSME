# Deterministic Method

The deterministic method computes transition probabilities between layers based on edge counts.

## Usage

```python
from lsme import LSME

lsme = LSME(
    method='deterministic',
    max_hops=3
)
result = lsme.fit_transform(G)

# Output dimension: 3 * (max_hops + 1) = 12
```

## API Reference

::: lsme.methods.DeterministicMethod
    options:
      show_root_heading: true
      show_source: true
