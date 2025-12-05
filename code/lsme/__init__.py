"""
LSME - Local Structural Matrix Embeddings for graphs.

A simple and efficient library for generating structural embeddings of nodes in graphs.
"""

from .lsme import LSME
from .core import compute_local_signature_matrix

__version__ = "0.1.0"
__all__ = ["LSME", "compute_local_signature_matrix"]