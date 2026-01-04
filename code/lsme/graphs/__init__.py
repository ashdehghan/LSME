"""Synthetic graph generators for LSME testing and experimentation."""

from .builder import SyntheticGraphBuilder
from .patterns import (
    build_random,
    build_barbell,
    build_web_pattern,
    build_star_pattern,
    build_dense_star,
    build_crossed_diamond,
    build_dynamic_star,
)

__all__ = [
    "SyntheticGraphBuilder",
    "build_random",
    "build_barbell",
    "build_web_pattern",
    "build_star_pattern",
    "build_dense_star",
    "build_crossed_diamond",
    "build_dynamic_star",
]
