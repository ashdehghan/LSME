"""CNN Autoencoder architecture for signature matrices."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class SignatureAutoencoder(nn.Module):
    """
    CNN Autoencoder for encoding signature matrices into fixed-size embeddings.

    Uses strided convolutions for downsampling (more efficient than pooling)
    and transposed convolutions for upsampling.

    Parameters
    ----------
    input_size : int, default=64
        Size of padded input matrices (assumed square).
    embedding_dim : int, default=32
        Dimension of bottleneck embedding.
    hidden_channels : list[int], optional
        Channel counts for each conv layer. Default is [32, 64, 128, 256].
    """

    def __init__(
        self,
        input_size: int = 64,
        embedding_dim: int = 32,
        hidden_channels: Optional[List[int]] = None
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels

        # Calculate spatial size after all convolutions
        # Each conv with stride=2 halves the spatial dimension
        self.final_spatial = input_size // (2 ** len(hidden_channels))
        self.flat_dim = hidden_channels[-1] * self.final_spatial ** 2

        # Build encoder
        self.encoder_conv = self._build_encoder_conv(hidden_channels)
        self.encoder_fc = nn.Linear(self.flat_dim, embedding_dim)

        # Build decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_dim, self.flat_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv = self._build_decoder_conv(hidden_channels)

    def _build_encoder_conv(self, channels: List[int]) -> nn.Sequential:
        """Build encoder convolutional layers."""
        layers = []
        in_channels = 1  # Single channel input (grayscale signature matrix)

        for out_channels in channels:
            layers.extend([
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _build_decoder_conv(self, channels: List[int]) -> nn.Sequential:
        """Build decoder convolutional layers."""
        layers = []
        reversed_channels = list(reversed(channels))

        for i, out_channels in enumerate(reversed_channels[1:]):
            in_channels = reversed_channels[i]
            layers.extend([
                nn.ConvTranspose2d(
                    in_channels, out_channels,
                    kernel_size=4, stride=2, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])

        # Final layer to single channel with sigmoid for [0, 1] output
        layers.extend([
            nn.ConvTranspose2d(
                reversed_channels[-1], 1,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
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
        x = self.encoder_conv(x)
        x = x.view(batch_size, -1)
        x = self.encoder_fc(x)
        return x

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
        x = self.decoder_fc(z)
        x = x.view(
            batch_size, self.hidden_channels[-1],
            self.final_spatial, self.final_spatial
        )
        x = self.decoder_conv(x)
        return x

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


def masked_mse_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute MSE loss only on non-padded regions.

    Parameters
    ----------
    reconstruction : torch.Tensor
        Reconstructed matrices, shape (batch, 1, H, W).
    target : torch.Tensor
        Original padded matrices, shape (batch, 1, H, W).
    mask : torch.Tensor
        Binary mask with 1s for valid regions, shape (batch, 1, H, W).

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    # Apply mask to both reconstruction and target
    masked_recon = reconstruction * mask
    masked_target = target * mask

    # Compute squared differences
    diff_squared = (masked_recon - masked_target) ** 2

    # Sum over spatial dimensions, normalize by valid element count per sample
    # Add small epsilon to avoid division by zero
    valid_counts = mask.sum(dim=[1, 2, 3]) + 1e-8
    loss_per_sample = diff_squared.sum(dim=[1, 2, 3]) / valid_counts

    return loss_per_sample.mean()
