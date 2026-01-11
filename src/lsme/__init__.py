"""
LSME - Local Structural Matrix Embeddings for graphs.

A library for generating structural embeddings of nodes in graphs
using multiple embedding methods.
"""

from .lsme import LSME
from .core import get_nodes_by_hop_distance
from .encoder import CNNEncoder, DNNEncoder, SignatureEncoder
from .graphs import SyntheticGraphBuilder

__version__ = "0.2.0"
__all__ = [
    "LSME",
    "CNNEncoder",
    "DNNEncoder",
    "SignatureEncoder",
    "SyntheticGraphBuilder",
    "get_nodes_by_hop_distance",
]
