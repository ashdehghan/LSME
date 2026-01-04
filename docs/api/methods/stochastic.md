# Stochastic Method

The stochastic method creates signature matrices through permutation averaging, then encodes them with a neural network.

## Usage

```python
from lsme import LSME

lsme = LSME(
    method='stochastic',
    max_hops=2,
    n_samples=100,
    embedding_dim=32,
    encoder_type='cnn'
)
result = lsme.fit_transform(G)
```

## API Reference

::: lsme.methods.StochasticMethod
    options:
      show_root_heading: true
      show_source: true
