"""Utility functions for signature matrix preprocessing."""

import warnings
import numpy as np


def pad_matrix(
    matrix: np.ndarray,
    target_size: int,
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Pad a square matrix to target size.

    Padding is added to the bottom and right to preserve the layer structure
    which starts from the top-left (root node at position 0,0).

    Parameters
    ----------
    matrix : np.ndarray
        Input square matrix of shape (n, n).
    target_size : int
        Target size for the output matrix.
    pad_value : float, default=0.0
        Value to use for padding (0.0 represents no structure).

    Returns
    -------
    np.ndarray
        Padded matrix of shape (target_size, target_size).

    Warns
    -----
    UserWarning
        If the matrix is larger than target_size and will be truncated.
    """
    original_size = matrix.shape[0]

    if original_size > target_size:
        # Truncate if larger than target - warn about data loss
        warnings.warn(
            f"Matrix of size {original_size}x{original_size} is larger than target size "
            f"{target_size}x{target_size}. Data will be truncated. "
            f"Consider increasing max_matrix_size parameter.",
            UserWarning
        )
        return matrix[:target_size, :target_size].astype(np.float32)

    if original_size == target_size:
        return matrix.astype(np.float32)

    padded = np.full((target_size, target_size), pad_value, dtype=np.float32)
    padded[:original_size, :original_size] = matrix

    return padded


def create_mask(original_size: int, target_size: int) -> np.ndarray:
    """
    Create binary mask for valid (non-padded) region.

    Parameters
    ----------
    original_size : int
        Original matrix dimension before padding.
    target_size : int
        Target padded size.

    Returns
    -------
    np.ndarray
        Binary mask of shape (target_size, target_size) with 1.0 for valid
        region and 0.0 for padded region.
    """
    mask = np.zeros((target_size, target_size), dtype=np.float32)
    effective_size = min(original_size, target_size)
    mask[:effective_size, :effective_size] = 1.0
    return mask


def compute_padded_size(
    max_original_size: int,
    max_allowed: int = 64,
    num_conv_layers: int = 4
) -> int:
    """
    Compute optimal padded size (power of 2) for efficient GPU operations.

    The minimum size is 2^num_conv_layers to ensure the spatial dimensions
    don't become smaller than 1 after all stride-2 convolutions.

    Parameters
    ----------
    max_original_size : int
        Maximum matrix size in the dataset.
    max_allowed : int, default=64
        Maximum allowed padded size.
    num_conv_layers : int, default=4
        Number of stride-2 conv layers in the encoder.

    Returns
    -------
    int
        Optimal padded size (power of 2, capped at max_allowed).
    Warns
    -----
    UserWarning
        If max_original_size > max_allowed, indicating data truncation will occur.
    """
    # Minimum size to ensure final_spatial >= 1 after all convolutions
    min_size = 2 ** num_conv_layers  # e.g., 16 for 4 layers

    # Round up to nearest power of 2
    size = max(min_size, max_original_size)
    padded = 2 ** int(np.ceil(np.log2(size)))

    # Warn if we're going to truncate data
    if padded > max_allowed:
        warnings.warn(
            f"Computed padded size {padded} exceeds max_allowed {max_allowed}. "
            f"Matrices larger than {max_allowed}x{max_allowed} will be truncated. "
            f"Consider increasing max_matrix_size parameter to preserve all data.",
            UserWarning
        )

    return min(padded, max_allowed)
