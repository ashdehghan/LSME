"""DNN Autoencoder architecture for signature matrices."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class SignatureDNN(nn.Module):
    """
    Dense Neural Network Autoencoder for encoding signature matrices.

    Flattens input matrices and uses fully-connected layers for encoding.
    Better suited for smaller matrices; CNN is generally better for larger ones.

    Parameters
    ----------
    input_size : int, default=64
        Size of padded input matrices (assumed square).
    embedding_dim : int, default=32
        Dimension of bottleneck embedding.
    hidden_dims : list[int], optional
        Dimensions for hidden layers. Default is [512, 256, 128].
    """

    def __init__(
        self,
        input_size: int = 64,
        embedding_dim: int = 32,
        hidden_dims: Optional[List[int]] = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.flat_dim = input_size * input_size

        # Build encoder
        self.encoder = self._build_encoder(hidden_dims)

        # Build decoder (mirror of encoder)
        self.decoder = self._build_decoder(hidden_dims)

    def _build_encoder(self, hidden_dims: List[int]) -> nn.Sequential:
        """Build encoder fully-connected layers."""
        layers = []
        in_dim = self.flat_dim

        for out_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = out_dim

        # Final layer to embedding dimension
        layers.append(nn.Linear(in_dim, self.embedding_dim))

        return nn.Sequential(*layers)

    def _build_decoder(self, hidden_dims: List[int]) -> nn.Sequential:
        """Build decoder fully-connected layers."""
        layers = []
        reversed_dims = list(reversed(hidden_dims))
        in_dim = self.embedding_dim

        for out_dim in reversed_dims:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = out_dim

        # Final layer to original flattened size with sigmoid
        layers.extend([
            nn.Linear(in_dim, self.flat_dim),
            nn.Sigmoid()
        ])

        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input matrices to embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape (batch, embedding_dim).
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings back to matrices.

        Parameters
        ----------
        z : torch.Tensor
            Embedding tensor of shape (batch, embedding_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape (batch, 1, H, W).
        """
        batch_size = z.shape[0]
        x = self.decoder(z)
        return x.view(batch_size, 1, self.input_size, self.input_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both reconstruction and embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        tuple
            (reconstruction, embedding) where reconstruction has the same
            shape as input and embedding has shape (batch, embedding_dim).
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, z
