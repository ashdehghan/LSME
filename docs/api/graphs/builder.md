# SyntheticGraphBuilder

Fluent builder for creating synthetic graphs from multiple patterns.

## Usage

```python
from lsme import SyntheticGraphBuilder

builder = SyntheticGraphBuilder(random_state=42)

# Chain multiple patterns
G = (builder
     .add_random(n_nodes=50, edge_prob=0.1)
     .add_barbell(count=3, m1=10, m2=5)
     .add_web_pattern(count=2, n_rings=2, spokes=4)
     .add_star_pattern(count=5, n_arms=4, arm_length=2)
     .hydrate(prob=0.05)  # Add random edges
     .build())

# Reset and build another
G2 = builder.reset().add_dense_star(count=10).build()
```

## API Reference

::: lsme.graphs.SyntheticGraphBuilder
    options:
      show_root_heading: true
      show_source: true
