"""PyTorch Dataset for LSME signature matrices."""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import pad_matrix, create_mask


class SignatureDataset(Dataset):
    """
    PyTorch Dataset for batched loading of padded signature matrices.

    Pre-computes padded matrices and masks on initialization for efficient
    training (trades memory for speed).

    Parameters
    ----------
    signature_matrices : dict
        Dict mapping node_id to 2D numpy arrays.
    layer_info : dict
        Dict mapping node_id to layer metadata (must contain 'total_nodes').
    max_size : int
        Size to pad all matrices to.
    """

    def __init__(
        self,
        signature_matrices: Dict[Any, np.ndarray],
        layer_info: Dict[Any, dict],
        max_size: int
    ):
        self.node_ids: List[Any] = list(signature_matrices.keys())
        self.max_size = max_size

        # Pre-compute all padded matrices and masks for efficiency
        self._matrices: List[np.ndarray] = []
        self._masks: List[np.ndarray] = []
        self._original_sizes: List[int] = []

        for nid in self.node_ids:
            matrix = signature_matrices[nid]
            orig_size = layer_info[nid]['total_nodes']

            padded = pad_matrix(matrix, max_size)
            mask = create_mask(orig_size, max_size)

            # Add channel dimension: (H, W) -> (1, H, W)
            self._matrices.append(padded[np.newaxis, :, :])
            self._masks.append(mask[np.newaxis, :, :])
            self._original_sizes.append(orig_size)

    def __len__(self) -> int:
        return len(self.node_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Any, int]:
        """
        Get a single sample.

        Returns
        -------
        tuple
            (padded_matrix, mask, node_id, original_size)
            - padded_matrix: torch.Tensor of shape (1, max_size, max_size)
            - mask: torch.Tensor of shape (1, max_size, max_size)
            - node_id: original node identifier
            - original_size: int, original matrix dimension
        """
        return (
            torch.from_numpy(self._matrices[idx]),
            torch.from_numpy(self._masks[idx]),
            self.node_ids[idx],
            self._original_sizes[idx]
        )


def collate_signature_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Any, int]]
) -> Tuple[torch.Tensor, torch.Tensor, List[Any], List[int]]:
    """
    Custom collate function for SignatureDataset.

    Parameters
    ----------
    batch : list
        List of (matrix, mask, node_id, original_size) tuples.

    Returns
    -------
    tuple
        (matrices, masks, node_ids, original_sizes) where matrices and masks
        are stacked tensors.
    """
    matrices = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    node_ids = [item[2] for item in batch]
    original_sizes = [item[3] for item in batch]

    return matrices, masks, node_ids, original_sizes
