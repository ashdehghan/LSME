"""
CNN Autoencoder for encoding LSME signature matrices into fixed-size embeddings.

This module provides the SignatureEncoder class for training a convolutional
autoencoder on signature matrices and extracting node embeddings.
"""

from .encoder import SignatureEncoder
from .dataset import SignatureDataset
from .model import SignatureAutoencoder, masked_mse_loss

__all__ = [
    "SignatureEncoder",
    "SignatureDataset",
    "SignatureAutoencoder",
    "masked_mse_loss"
]
