"""CNN Encoder class for encoding LSME signature matrices."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .base import BaseEncoder
from .dataset import SignatureDataset, collate_signature_batch
from .model import SignatureAutoencoder, masked_mse_loss
from .utils import compute_padded_size, pad_matrix, create_mask


class CNNEncoder(BaseEncoder):
    """
    CNN Autoencoder for encoding LSME signature matrices into fixed-size embeddings.

    Uses strided convolutions for efficient encoding of 2D signature matrices.

    Parameters
    ----------
    embedding_dim : int, default=32
        Dimension of the output embedding vector.

    max_matrix_size : int, default=64
        Maximum matrix size to pad to. Matrices larger than this are truncated.

    hidden_channels : list[int], optional
        Number of channels in each conv layer. Default is [32, 64, 128, 256].

    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer.

    batch_size : int, default=32
        Batch size for training.

    num_epochs : int, default=100
        Number of training epochs.

    device : str, default='auto'
        Device to use ('cuda', 'cpu', or 'auto' for automatic detection).

    verbose : bool, default=True
        Whether to print progress information.

    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> from lsme import LSME
    >>> from lsme.encoder import CNNEncoder
    >>> import networkx as nx
    >>>
    >>> G = nx.karate_club_graph()
    >>> lsme = LSME(max_hops=2, n_samples=100)
    >>> result = lsme.fit_transform(G)
    >>>
    >>> encoder = CNNEncoder(embedding_dim=32, num_epochs=50)
    >>> embeddings = encoder.fit_transform(
    ...     result['signature_matrices'],
    ...     result['layer_info']
    ... )
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        max_matrix_size: int = 64,
        hidden_channels: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        num_epochs: int = 100,
        device: str = 'auto',
        verbose: bool = True,
        random_state: Optional[int] = None
    ):
        self.embedding_dim = embedding_dim
        self.max_matrix_size = max_matrix_size
        self.hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device_str = device
        self.verbose = verbose
        self.random_state = random_state

        self._model: Optional[SignatureAutoencoder] = None
        self._device: Optional[torch.device] = None
        self._padded_size: Optional[int] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Check if the encoder has been fitted."""
        return self._is_fitted

    def _get_device(self) -> torch.device:
        """Determine the device to use."""
        if self.device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device_str)

    def _set_random_state(self) -> None:
        """Set random seeds for reproducibility."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Run a single training epoch."""
        self._model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            matrices, masks, _, _ = batch
            matrices = matrices.to(self._device, non_blocking=True)
            masks = masks.to(self._device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            reconstruction, _ = self._model(matrices)
            loss = masked_mse_loss(reconstruction, matrices, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _validate(self, dataloader: DataLoader) -> float:
        """Run validation and return average loss."""
        self._model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                matrices, masks, _, _ = batch
                matrices = matrices.to(self._device, non_blocking=True)
                masks = masks.to(self._device, non_blocking=True)

                reconstruction, _ = self._model(matrices)
                loss = masked_mse_loss(reconstruction, matrices, masks)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def fit(
        self,
        signature_matrices: Dict[Any, np.ndarray],
        layer_info: Dict[Any, dict],
        validation_split: float = 0.1
    ) -> 'CNNEncoder':
        """
        Train the autoencoder on signature matrices.

        Parameters
        ----------
        signature_matrices : dict
            Dict mapping node_id to 2D numpy arrays (from LSME.fit_transform).
        layer_info : dict
            Dict mapping node_id to layer metadata (from LSME.fit_transform).
        validation_split : float, default=0.1
            Fraction of data for validation.

        Returns
        -------
        self
            Fitted encoder.
        """
        self._set_random_state()
        self._device = self._get_device()

        n_samples = len(signature_matrices)
        if n_samples == 0:
            raise ValueError("Cannot fit encoder on empty dataset")

        if self.verbose:
            print(f"Training CNNEncoder on device: {self._device}")

        # Defensive check: encoder requires at least 2 samples for BatchNorm
        # This should not be reached if LSME validation is working correctly
        if n_samples < 2:
            raise ValueError(
                f"Encoder requires at least 2 samples for training (BatchNorm constraint). "
                f"Got {n_samples} sample(s). Use LSME with at least 3 nodes for stochastic method."
            )

        # Determine optimal padded size
        max_original = max(info['total_nodes'] for info in layer_info.values())
        self._padded_size = compute_padded_size(
            max_original, self.max_matrix_size, len(self.hidden_channels)
        )

        if self.verbose:
            print(f"Max matrix size in data: {max_original}")
            print(f"Padded size: {self._padded_size}x{self._padded_size}")

        # Create dataset
        full_dataset = SignatureDataset(
            signature_matrices, layer_info, self._padded_size
        )

        # Train/validation split
        # Skip validation for very small datasets to avoid BatchNorm issues
        # BatchNorm requires batch_size > 1, so we need at least 2 training samples
        n_total = len(full_dataset)
        min_train_samples = 2  # Minimum for BatchNorm to work

        if n_total < min_train_samples + 1:
            # Too few samples for validation split
            n_val = 0
            n_train = n_total
        else:
            n_val = max(1, int(n_total * validation_split))
            # Ensure we keep at least min_train_samples for training
            if n_total - n_val < min_train_samples:
                n_val = max(0, n_total - min_train_samples)
            n_train = n_total - n_val

        if n_val > 0:
            train_dataset, val_dataset = random_split(
                full_dataset, [n_train, n_val]
            )
        else:
            train_dataset = full_dataset
            val_dataset = None

        # Create dataloaders
        use_pin_memory = self._device.type == 'cuda'
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_pin_memory,
            collate_fn=collate_signature_batch,
            drop_last=False
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=use_pin_memory,
                collate_fn=collate_signature_batch
            )

        # Initialize model
        self._model = SignatureAutoencoder(
            input_size=self._padded_size,
            embedding_dim=self.embedding_dim,
            hidden_channels=self.hidden_channels
        ).to(self._device)

        if self.verbose:
            n_params = sum(p.numel() for p in self._model.parameters())
            print(f"Model parameters: {n_params:,}")

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        # Training loop
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(train_loader, optimizer)

            if val_loader is not None:
                val_loss = self._validate(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self._model.state_dict().items()
                    }

                if self.verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{self.num_epochs} - "
                        f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )
            else:
                scheduler.step(train_loss)
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.6f}")

        # Restore best model if we had validation
        if best_model_state is not None:
            self._model.load_state_dict(best_model_state)

        self._is_fitted = True

        if self.verbose:
            print("Training complete!")

        return self

    def encode(
        self,
        signature_matrices: Dict[Any, np.ndarray],
        layer_info: Optional[Dict[Any, dict]] = None
    ) -> Dict[Any, np.ndarray]:
        """
        Encode signature matrices into fixed-size embeddings.

        Parameters
        ----------
        signature_matrices : dict
            Dict mapping node_id to 2D numpy arrays.
        layer_info : dict, optional
            Layer metadata (used for efficient padding). If not provided,
            original sizes are inferred from matrix dimensions.

        Returns
        -------
        dict
            Dict mapping node_id to 1D embedding numpy array of shape (embedding_dim,).
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder must be fitted before encoding. Call fit() first.")

        self._model.eval()
        embeddings: Dict[Any, np.ndarray] = {}

        with torch.no_grad():
            for node_id, matrix in signature_matrices.items():
                # Get original size
                if layer_info is not None:
                    orig_size = layer_info[node_id]['total_nodes']
                else:
                    orig_size = matrix.shape[0]

                # Pad and add batch/channel dimensions
                padded = pad_matrix(matrix, self._padded_size)
                tensor = torch.from_numpy(padded[np.newaxis, np.newaxis, :, :])
                tensor = tensor.to(self._device)

                # Encode
                embedding = self._model.encode(tensor)
                embeddings[node_id] = embedding.cpu().numpy().squeeze()

        return embeddings

    def decode(self, embeddings: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        """
        Decode embeddings back to signature matrices.

        Parameters
        ----------
        embeddings : dict
            Dict mapping node_id to 1D embedding numpy arrays.

        Returns
        -------
        dict
            Dict mapping node_id to reconstructed 2D numpy arrays
            (padded size, values in [0, 1]).
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder must be fitted before decoding. Call fit() first.")

        self._model.eval()
        reconstructed: Dict[Any, np.ndarray] = {}

        with torch.no_grad():
            for node_id, emb in embeddings.items():
                # Add batch dimension
                tensor = torch.from_numpy(emb[np.newaxis, :]).float()
                tensor = tensor.to(self._device)

                # Decode
                recon = self._model.decode(tensor)
                reconstructed[node_id] = recon.cpu().numpy().squeeze()

        return reconstructed

    def fit_transform(
        self,
        signature_matrices: Dict[Any, np.ndarray],
        layer_info: Dict[Any, dict],
        validation_split: float = 0.1
    ) -> Dict[Any, np.ndarray]:
        """
        Train the autoencoder and return embeddings.

        Parameters
        ----------
        signature_matrices : dict
            Dict mapping node_id to 2D numpy arrays.
        layer_info : dict
            Dict mapping node_id to layer metadata.
        validation_split : float, default=0.1
            Fraction of data for validation.

        Returns
        -------
        dict
            Dict mapping node_id to 1D embedding numpy arrays.
        """
        self.fit(signature_matrices, layer_info, validation_split)
        return self.encode(signature_matrices, layer_info)

    def reconstruction_error(
        self,
        signature_matrices: Dict[Any, np.ndarray],
        layer_info: Optional[Dict[Any, dict]] = None
    ) -> Dict[Any, float]:
        """
        Compute per-node reconstruction error (MSE on valid region).

        Parameters
        ----------
        signature_matrices : dict
            Dict mapping node_id to 2D numpy arrays.
        layer_info : dict, optional
            Layer metadata for computing masked error.

        Returns
        -------
        dict
            Dict mapping node_id to reconstruction MSE (float).
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder must be fitted first.")

        embeddings = self.encode(signature_matrices, layer_info)
        reconstructed = self.decode(embeddings)

        errors: Dict[Any, float] = {}
        for node_id, matrix in signature_matrices.items():
            if layer_info is not None:
                orig_size = layer_info[node_id]['total_nodes']
            else:
                orig_size = matrix.shape[0]

            orig_padded = pad_matrix(matrix, self._padded_size)
            recon = reconstructed[node_id]
            mask = create_mask(orig_size, self._padded_size)

            # Compute MSE only on valid region
            diff_sq = (orig_padded - recon) ** 2
            mse = (diff_sq * mask).sum() / mask.sum()
            errors[node_id] = float(mse)

        return errors

    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        path : str
            Path to save the model (typically .pt extension).
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted encoder.")

        state = {
            'encoder_type': 'cnn',
            'model_state_dict': self._model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'max_matrix_size': self.max_matrix_size,
            'hidden_channels': self.hidden_channels,
            'padded_size': self._padded_size,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'random_state': self.random_state
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'CNNEncoder':
        """
        Load a trained model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model.
        device : str, default='auto'
            Device to load the model to.

        Returns
        -------
        CNNEncoder
            Loaded encoder ready for encoding.
        """
        state = torch.load(path, map_location='cpu')

        encoder = cls(
            embedding_dim=state['embedding_dim'],
            max_matrix_size=state['max_matrix_size'],
            hidden_channels=state['hidden_channels'],
            learning_rate=state['learning_rate'],
            batch_size=state['batch_size'],
            num_epochs=state['num_epochs'],
            device=device,
            random_state=state['random_state']
        )

        encoder._padded_size = state['padded_size']
        encoder._device = encoder._get_device()

        encoder._model = SignatureAutoencoder(
            input_size=encoder._padded_size,
            embedding_dim=encoder.embedding_dim,
            hidden_channels=encoder.hidden_channels
        ).to(encoder._device)

        encoder._model.load_state_dict(state['model_state_dict'])
        encoder._is_fitted = True

        return encoder


# Backward-compatible alias
SignatureEncoder = CNNEncoder
