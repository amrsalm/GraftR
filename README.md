# graphtR <img src="https://img.shields.io/badge/dev-0.1.0-orange" align="right" />

An experimental R package for learning on graph-structured data using **Graph Neural Networks (GNNs)**.  
The goal of this package is to bring foundational GNN models to R, beginning with a **Variational Graph Autoencoder (VGAE)** built entirely from scratch using base R.

---

## 🚧 Status: Under Development

This package is in active development. The current version supports Variational Graph Autoencoders (VGAE). Future updates will include Graph Convolutional Networks (GCN), Message Passing Neural Networks (MPNN), and potentially Graph Attention Networks (GAT).

---

## ✨ Features

- ✅ Fully functioning VGAE implemented in base R  
- ✅ Numerical gradient-based training
- ✅ Matrix-based encoder/decoder pipeline
- ✅ Support for graph reconstruction and link prediction
- 📊 Output includes learned node embeddings and adjacency predictions

---

## 📘 What is a VGAE?

A **Variational Graph Autoencoder (VGAE)** is an unsupervised graph neural network model that learns **latent embeddings for each node** such that it can reconstruct the original graph.

It consists of:

1. **Encoder** – A Graph Convolutional Network (GCN) that computes a probabilistic distribution over node embeddings (mean and variance).
2. **Reparameterization Trick** – Samples embeddings from the learned distribution.
3. **Decoder** – Reconstructs the adjacency matrix using an inner product between latent node vectors.
4. **Training Objective (ELBO)** – Maximizes the likelihood of reconstructing edges while regularizing the embedding space via KL divergence.

---

## 🔧 Installation

Since this is not yet on CRAN, you can install it from source:

```r
# Not yet available. For now, clone manually:
# git clone https://github.com/amrsalm/GraftR.git

# In R
devtools::load_all("path/to/gRaphnet")
