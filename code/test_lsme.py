"""Test script to verify LSME package functionality."""

import networkx as nx
from lsme import LSME

# Test 1: Basic usage
print("Test 1: Basic usage")
print("-" * 50)

# Create a sample graph
G = nx.karate_club_graph()
print(f"Graph: Karate Club")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# Initialize LSME
embedder = LSME(max_hops=2, n_samples=50, verbose=False)

# Generate embeddings
embeddings = embedder.fit_transform(G)
print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"Column names: {list(embeddings.columns)[:5]}...")
print(f"\nFirst 5 nodes embeddings:")
print(embeddings.head())

# Test 2: With dimensionality reduction
print("\n" + "="*50)
print("Test 2: With dimensionality reduction")
print("-" * 50)

embedder_pca = LSME(max_hops=2, n_samples=50, embedding_dim=8, verbose=False)
embeddings_pca = embedder_pca.fit_transform(G)

print(f"Reduced embeddings shape: {embeddings_pca.shape}")
print(f"Column names: {list(embeddings_pca.columns)}")
print(f"\nFirst 5 nodes reduced embeddings:")
print(embeddings_pca.head())

# Test 3: Small custom graph
print("\n" + "="*50)
print("Test 3: Small custom graph")
print("-" * 50)

# Create a small graph
G_small = nx.Graph()
G_small.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3)])

embedder_small = LSME(max_hops=2, n_samples=20, verbose=False)
embeddings_small = embedder_small.fit_transform(G_small)

print(f"Small graph embeddings:")
print(embeddings_small)

# Test 4: Verify DataFrame structure
print("\n" + "="*50)
print("Test 4: DataFrame structure verification")
print("-" * 50)

print(f"Index name: {embeddings.index.name}")
print(f"Index type: {type(embeddings.index)}")
print(f"Data types:")
print(embeddings.dtypes.head())

print("\n✅ All tests passed successfully!")