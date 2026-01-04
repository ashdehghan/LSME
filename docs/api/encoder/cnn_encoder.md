# CNNEncoder

CNN-based autoencoder for encoding signature matrices.

## Usage

```python
from lsme import LSME, CNNEncoder

# Generate signature matrices
lsme = LSME(method='stochastic', max_hops=2)
result = lsme.fit_transform(G)

# Create and train encoder
encoder = CNNEncoder(
    embedding_dim=32,
    hidden_channels=[32, 64, 128, 256],
    num_epochs=100,
    learning_rate=1e-3
)

embeddings = encoder.fit_transform(
    result['signature_matrices'],
    result['layer_info']
)

# Encode new data
new_embeddings = encoder.encode(new_matrices, new_layer_info)

# Reconstruct
reconstructed = encoder.decode(embeddings)
```

## API Reference

::: lsme.encoder.CNNEncoder
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
