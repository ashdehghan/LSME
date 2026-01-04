# Graphs Overview

The graphs subpackage provides utilities for creating synthetic graphs with known structural patterns.

## Main Class

| Class | Description |
|-------|-------------|
| [`SyntheticGraphBuilder`](builder.md) | Fluent builder for composing graphs |

## Pattern Functions

| Function | Description |
|----------|-------------|
| `build_random()` | Erdos-Renyi random graph |
| `build_barbell()` | Two cliques connected by path |
| `build_web_pattern()` | Hub with concentric rings |
| `build_star_pattern()` | Central hub with arms |
| `build_dense_star()` | Simple star graph |
| `build_crossed_diamond()` | Diamond with crossed center |
| `build_dynamic_star()` | Star with configurable size |

## Quick Usage

```python
from lsme import SyntheticGraphBuilder

# Fluent builder
G = (SyntheticGraphBuilder(random_state=42)
     .add_random(n_nodes=100, edge_prob=0.1)
     .add_barbell(count=3)
     .add_star_pattern(count=5)
     .hydrate(prob=0.05)
     .build())

# Direct pattern functions
from lsme.graphs import build_barbell, build_web_pattern

G1 = build_barbell(m1=10, m2=10)
G2 = build_web_pattern(n_rings=3, spokes=6)
```
