# Encoders Overview

The encoder subpackage provides neural network autoencoders for encoding variable-sized signature matrices into fixed-size embeddings.

## Available Encoders

| Encoder | Architecture | Best For |
|---------|-------------|----------|
| [`CNNEncoder`](cnn_encoder.md) | Convolutional | Larger matrices, spatial patterns |
| [`DNNEncoder`](dnn_encoder.md) | Fully-connected | Smaller matrices, simpler patterns |

## Quick Usage

```python
from lsme import LSME, CNNEncoder, DNNEncoder

# Option 1: Integrated with LSME
lsme = LSME(
    method='stochastic',
    encoder_type='cnn',  # or 'dnn'
    embedding_dim=32
)
result = lsme.fit_transform(G)

# Option 2: Standalone encoder
lsme = LSME(method='stochastic')
result = lsme.fit_transform(G)

encoder = CNNEncoder(embedding_dim=64)
embeddings = encoder.fit_transform(
    result['signature_matrices'],
    result['layer_info']
)
```

## Common Interface

Both encoders share the same interface:

```python
# Training
encoder.fit(signature_matrices, layer_info)

# Encoding
embeddings = encoder.encode(signature_matrices, layer_info)

# Decoding (reconstruction)
reconstructed = encoder.decode(embeddings)

# Combined
embeddings = encoder.fit_transform(signature_matrices, layer_info)

# Evaluation
errors = encoder.reconstruction_error(signature_matrices, layer_info)

# Persistence
encoder.save('encoder.pt')
encoder = CNNEncoder.load('encoder.pt')
```
