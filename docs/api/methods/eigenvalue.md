# Eigenvalue Method

The eigenvalue method extracts eigenvalues from layer transition matrices.

## Usage

```python
from lsme import LSME

lsme = LSME(
    method='eigenvalue',
    max_hops=3
)
result = lsme.fit_transform(G)

# Output dimension: max_hops + 1 = 4
```

## API Reference

::: lsme.methods.EigenvalueMethod
    options:
      show_root_heading: true
      show_source: true
