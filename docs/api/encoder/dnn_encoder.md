# DNNEncoder

Dense neural network autoencoder for encoding signature matrices.

## Usage

```python
from lsme import LSME, DNNEncoder

# Generate signature matrices
lsme = LSME(method='stochastic', max_hops=2)
result = lsme.fit_transform(G)

# Create and train encoder
encoder = DNNEncoder(
    embedding_dim=32,
    hidden_dims=[512, 256, 128],
    num_epochs=100,
    learning_rate=1e-3
)

embeddings = encoder.fit_transform(
    result['signature_matrices'],
    result['layer_info']
)
```

## API Reference

::: lsme.encoder.DNNEncoder
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - fit
        - encode
        - decode
        - fit_transform
        - reconstruction_error
        - save
        - load
