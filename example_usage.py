#!/usr/bin/env python3
"""Example usage of CrystalFramer Encoder"""

import numpy as np
import crystalframer_encoder as cfe
from pymatgen.core import Structure, Lattice
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load the pretrained encoder
print("Loading pretrained CrystalFramer encoder...")
encoder = cfe.CrystalEncoder.from_pretrained(
    "./crystalframer_weight/formation_energy/best.ckpt"
)
print(f"Encoder loaded. Embedding dimension: {encoder.get_embedding_dim()}")

# Create some example structures
structures = []

# 1. Simple cubic Si
lattice = Lattice.cubic(5.43)
si = Structure(
    lattice,
    ["Si", "Si", "Si", "Si", "Si", "Si", "Si", "Si"],
    [[0.0, 0.0, 0.0],
     [0.25, 0.25, 0.25],
     [0.5, 0.5, 0.0],
     [0.75, 0.75, 0.25],
     [0.5, 0.0, 0.5],
     [0.75, 0.25, 0.75],
     [0.0, 0.5, 0.5],
     [0.25, 0.75, 0.75]]
)
structures.append(si)

# 2. Face-centered cubic Al
lattice = Lattice.cubic(4.05)
al = Structure(
    lattice,
    ["Al", "Al", "Al", "Al"],
    [[0.0, 0.0, 0.0],
     [0.5, 0.5, 0.0],
     [0.5, 0.0, 0.5],
     [0.0, 0.5, 0.5]]
)
structures.append(al)

# 3. Simple cubic Po
lattice = Lattice.cubic(3.35)
po = Structure(lattice, ["Po"], [[0.0, 0.0, 0.0]])
structures.append(po)

# 4. Diamond C
lattice = Lattice.cubic(3.57)
c = Structure(
    lattice,
    ["C", "C", "C", "C", "C", "C", "C", "C"],
    [[0.0, 0.0, 0.0],
     [0.25, 0.25, 0.25],
     [0.5, 0.5, 0.0],
     [0.75, 0.75, 0.25],
     [0.5, 0.0, 0.5],
     [0.75, 0.25, 0.75],
     [0.0, 0.5, 0.5],
     [0.25, 0.75, 0.75]]
)
structures.append(c)

# Encode all structures
print("\nEncoding structures...")
embeddings = encoder.encode(structures, batch_size=2, show_progress=True)
print(f"Embeddings shape: {embeddings.shape}")

# Test single structure encoding
print("\nSingle structure encoding test:")
single_embedding = encoder.encode_single(structures[0])
print(f"Single embedding shape: {single_embedding.shape}")

# Compute similarity matrix
print("\nComputing similarity matrix...")
similarity_matrix = encoder.compute_similarity(structures)
print("Similarity matrix (cosine):")
print(similarity_matrix.numpy())

# Find similar structures
print("\nFinding structures similar to Si...")
similar_to_si = encoder.find_similar_structures(
    structures[0], structures[1:], top_k=3, metric='cosine'
)
for idx, score in similar_to_si:
    print(f"  Structure {idx+1}: {structures[idx+1].formula} - similarity = {score:.3f}")

# Visualize embeddings using t-SNE
print("\nVisualizing embeddings with t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=2)
embeddings_2d = tsne.fit_transform(embeddings.numpy())

plt.figure(figsize=(8, 6))
labels = [s.formula for s in structures]
for i, label in enumerate(labels):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], s=100)
    plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2') 
plt.title('Crystal Structure Embeddings')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./embeddings_visualization.png', dpi=300)
print("Visualization saved to embeddings_visualization.png")

# # Save encoder for later use
# print("\nSaving encoder...")
# encoder.save_encoder("./my_encoder.pth")

# # Load and test
# print("Loading saved encoder...")
# loaded_encoder = cfe.CrystalEncoder.load_encoder("./my_encoder.pth")
# test_embedding = loaded_encoder.encode_single(structures[0])
# print(f"Test embedding from loaded encoder: {test_embedding.shape}")

# print("\nExample completed successfully!")