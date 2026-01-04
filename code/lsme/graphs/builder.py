"""Synthetic graph builder for composing complex graphs from patterns."""

import random
from typing import Dict, List, Optional, Any

import networkx as nx

from .patterns import (
    build_random,
    build_barbell,
    build_web_pattern,
    build_star_pattern,
    build_dense_star,
    build_crossed_diamond,
    build_dynamic_star,
)


class SyntheticGraphBuilder:
    """
    Builder for creating synthetic graphs by combining multiple patterns.

    Supports fluent interface for chaining operations.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> builder = SyntheticGraphBuilder(random_state=42)
    >>> G = (builder
    ...     .add_random(n_nodes=100, edge_prob=0.1)
    ...     .add_barbell(count=5)
    ...     .add_web_pattern(count=10)
    ...     .hydrate(prob=0.05)
    ...     .build())
    >>> G.number_of_nodes()
    250
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self._graphs: List[nx.Graph] = []
        self._hydration_config: Optional[Dict[str, Any]] = None
        self._trim_prob: Optional[float] = None

        if random_state is not None:
            random.seed(random_state)

    def add_random(self, n_nodes: int, edge_prob: float) -> 'SyntheticGraphBuilder':
        """
        Add a random Erdos-Renyi graph.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.
        edge_prob : float
            Edge probability.

        Returns
        -------
        self
        """
        G = build_random(n_nodes, edge_prob, seed=self.random_state)
        self._graphs.append(G)
        return self

    def add_barbell(self, count: int = 1, m1: int = 10, m2: int = 10) -> 'SyntheticGraphBuilder':
        """
        Add barbell graph(s).

        Parameters
        ----------
        count : int, default=1
            Number of barbell graphs to add.
        m1 : int, default=10
            Nodes per clique.
        m2 : int, default=10
            Nodes in connecting path.

        Returns
        -------
        self
        """
        for _ in range(count):
            G = build_barbell(m1, m2)
            self._graphs.append(G)
        return self

    def add_web_pattern(
        self,
        count: int = 1,
        n_rings: int = 2,
        spokes: int = 5
    ) -> 'SyntheticGraphBuilder':
        """
        Add web pattern graph(s).

        Parameters
        ----------
        count : int, default=1
            Number of web patterns to add.
        n_rings : int, default=2
            Number of concentric rings.
        spokes : int, default=5
            Number of radial spokes.

        Returns
        -------
        self
        """
        for _ in range(count):
            G = build_web_pattern(n_rings, spokes)
            self._graphs.append(G)
        return self

    def add_star_pattern(
        self,
        count: int = 1,
        n_arms: int = 5,
        arm_length: int = 2
    ) -> 'SyntheticGraphBuilder':
        """
        Add star pattern graph(s).

        Parameters
        ----------
        count : int, default=1
            Number of star patterns to add.
        n_arms : int, default=5
            Number of arms.
        arm_length : int, default=2
            Length of each arm.

        Returns
        -------
        self
        """
        for _ in range(count):
            G = build_star_pattern(n_arms, arm_length)
            self._graphs.append(G)
        return self

    def add_dense_star(self, count: int = 1, n_leaves: int = 12) -> 'SyntheticGraphBuilder':
        """
        Add dense star graph(s).

        Parameters
        ----------
        count : int, default=1
            Number of dense stars to add.
        n_leaves : int, default=12
            Number of leaves per star.

        Returns
        -------
        self
        """
        for _ in range(count):
            G = build_dense_star(n_leaves)
            self._graphs.append(G)
        return self

    def add_crossed_diamond(self, count: int = 1) -> 'SyntheticGraphBuilder':
        """
        Add crossed diamond graph(s).

        Parameters
        ----------
        count : int, default=1
            Number of crossed diamonds to add.

        Returns
        -------
        self
        """
        for _ in range(count):
            G = build_crossed_diamond()
            self._graphs.append(G)
        return self

    def add_dynamic_star(self, count: int = 1, n_leaves: int = 5) -> 'SyntheticGraphBuilder':
        """
        Add dynamic star graph(s).

        Parameters
        ----------
        count : int, default=1
            Number of dynamic stars to add.
        n_leaves : int, default=5
            Number of leaves.

        Returns
        -------
        self
        """
        for _ in range(count):
            G = build_dynamic_star(n_leaves)
            self._graphs.append(G)
        return self

    def hydrate(
        self,
        prob: float = 0.1,
        method: str = 'standard'
    ) -> 'SyntheticGraphBuilder':
        """
        Configure hydration (adding random edges after combining).

        Parameters
        ----------
        prob : float, default=0.1
            Probability of adding edge between eligible node pairs.
        method : str, default='standard'
            Hydration method:
            - 'standard': Add random edges between any nodes
            - 'bridge_connect': Preferentially connect bridge nodes

        Returns
        -------
        self
        """
        self._hydration_config = {'prob': prob, 'method': method}
        return self

    def trim(self, prob: float = 0.1) -> 'SyntheticGraphBuilder':
        """
        Configure trimming (removing random edges after combining).

        Parameters
        ----------
        prob : float, default=0.1
            Probability of removing each edge.

        Returns
        -------
        self
        """
        self._trim_prob = prob
        return self

    def _relabel_graphs(self, graphs: List[nx.Graph]) -> List[nx.Graph]:
        """Relabel nodes in graphs to ensure globally unique node IDs."""
        relabeled = []
        offset = 0

        for G in graphs:
            mapping = {node: node + offset for node in G.nodes()}
            relabeled.append(nx.relabel_nodes(G, mapping))
            offset += G.number_of_nodes()

        return relabeled

    def _combine_graphs(self, graphs: List[nx.Graph]) -> nx.Graph:
        """Combine multiple graphs into one."""
        if not graphs:
            return nx.Graph()

        combined = graphs[0].copy()
        for G in graphs[1:]:
            combined = nx.compose(combined, G)

        return combined

    def _apply_hydration(self, G: nx.Graph) -> nx.Graph:
        """Apply hydration to add random edges."""
        if self._hydration_config is None:
            return G

        prob = self._hydration_config['prob']
        method = self._hydration_config['method']

        nodes = list(G.nodes())

        if method == 'bridge_connect':
            # Preferentially connect bridge nodes
            bridge_nodes = [
                n for n in nodes
                if G.nodes[n].get('con_type') == 'bridge'
                and G.nodes[n].get('role') != 'r'
            ]
            target_nodes = [
                n for n in nodes
                if G.nodes[n].get('con_type') == 'bridge'
            ]

            for node in bridge_nodes:
                if random.random() < prob and target_nodes:
                    target = random.choice(target_nodes)
                    if node != target and not G.has_edge(node, target):
                        G.add_edge(node, target)
        else:
            # Standard: add random edges to non-random nodes
            non_random_nodes = [
                n for n in nodes if G.nodes[n].get('role') != 'r'
            ]

            for node in non_random_nodes:
                if random.random() < prob:
                    target = random.choice(nodes)
                    if node != target and not G.has_edge(node, target):
                        G.add_edge(node, target)

        return G

    def _apply_trim(self, G: nx.Graph) -> nx.Graph:
        """Apply trimming to remove random edges."""
        if self._trim_prob is None:
            return G

        edges_to_remove = [
            edge for edge in G.edges()
            if random.random() < self._trim_prob
        ]

        G.remove_edges_from(edges_to_remove)
        return G

    def _extract_largest_component(self, G: nx.Graph) -> nx.Graph:
        """Extract the largest connected component."""
        if G.number_of_nodes() == 0:
            return G

        components = list(nx.connected_components(G))
        if not components:
            return G

        largest = max(components, key=len)
        return G.subgraph(largest).copy()

    def _reset_node_labels(self, G: nx.Graph) -> nx.Graph:
        """Reset node labels to sequential integers starting from 0."""
        mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
        return nx.relabel_nodes(G, mapping)

    def build(self) -> nx.Graph:
        """
        Build the final graph.

        Combines all added patterns, applies hydration and trimming,
        extracts largest connected component, and resets node labels.

        Returns
        -------
        nx.Graph
            The constructed graph with node attributes preserved.
        """
        if not self._graphs:
            return nx.Graph()

        # Relabel and combine
        relabeled = self._relabel_graphs(self._graphs)
        combined = self._combine_graphs(relabeled)

        # Apply modifications
        combined = self._apply_hydration(combined)
        combined = self._apply_trim(combined)

        # Extract LCC and reset labels
        result = self._extract_largest_component(combined)
        result = self._reset_node_labels(result)

        return result

    def reset(self) -> 'SyntheticGraphBuilder':
        """
        Reset the builder to initial state.

        Returns
        -------
        self
        """
        self._graphs = []
        self._hydration_config = None
        self._trim_prob = None
        return self
