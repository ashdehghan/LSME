"""Abstract base class for signature matrix encoders."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseEncoder(ABC):
    """
    Abstract base class for signature matrix encoders.

    All encoder implementations must inherit from this class and implement
    the required abstract methods.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the encoder has been fitted."""
        pass

    @abstractmethod
    def fit(
        self,
        signature_matrices: Dict[Any, np.ndarray],
        layer_info: Dict[Any, dict],
        validation_split: float = 0.1
    ) -> 'BaseEncoder':
        """
        Train the encoder on signature matrices.

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
        self
            Fitted encoder.
        """
        pass

    @abstractmethod
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
            Layer metadata for efficient padding.

        Returns
        -------
        dict
            Dict mapping node_id to 1D embedding numpy arrays.
        """
        pass

    @abstractmethod
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
            Dict mapping node_id to reconstructed 2D numpy arrays.
        """
        pass

    @abstractmethod
    def fit_transform(
        self,
        signature_matrices: Dict[Any, np.ndarray],
        layer_info: Dict[Any, dict],
        validation_split: float = 0.1
    ) -> Dict[Any, np.ndarray]:
        """
        Train the encoder and return embeddings.

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
        pass

    @abstractmethod
    def reconstruction_error(
        self,
        signature_matrices: Dict[Any, np.ndarray],
        layer_info: Optional[Dict[Any, dict]] = None
    ) -> Dict[Any, float]:
        """
        Compute per-node reconstruction error.

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
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained encoder to disk.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: str = 'auto') -> 'BaseEncoder':
        """
        Load a trained encoder from disk.

        Parameters
        ----------
        path : str
            Path to the saved model.
        device : str, default='auto'
            Device to load the model to.

        Returns
        -------
        BaseEncoder
            Loaded encoder ready for encoding.
        """
        pass
