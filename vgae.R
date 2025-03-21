#' Train a Variational Graph Autoencoder (VGAE)
#'
#' This function trains a Variational Graph Autoencoder (VGAE) on an input graph
#' represented by its adjacency matrix and node features. It uses a two-layer
#' GCN encoder, the reparameterization trick, and an inner product decoder to reconstruct
#' the graph structure.
#'
#' @param A A binary adjacency matrix of size \eqn{N \times N} representing the graph structure.
#' @param X A numeric feature matrix of size \eqn{N \times d} containing node features.
#' @param num_epochs Integer. Number of training epochs. Default is 10.
#' @param learning_rate Numeric. Learning rate for gradient ascent. Default is 1e-8.
#' @param verbose Logical. If TRUE, prints ELBO at each epoch. Default is TRUE.
#'
#' @return A list containing:
#' \describe{
#'   \item{embeddings}{The learned node embeddings \eqn{Z \in \mathbb{R}^{N \times d}}.}
#'   \item{A_pred}{The reconstructed adjacency matrix from the decoder.}
#'   \item{mu}{The mean matrix from the encoder.}
#'   \item{log_var}{The log-variance matrix from the encoder.}
#'   \item{weights}{A list of learned weights: \code{W}, \code{W_mu}, \code{W_0}.}
#' }
#' @export
#'
#' @examples
#' \dontrun{
#' data <- read.csv("absenteeism.csv", sep = ";", header = TRUE)
#' # preprocess adjacency matrix A and feature matrix X
#' result <- vgae(A, X)
#' }

vgae <- function(A, X, num_epochs = 10, learning_rate = 1e-8, verbose = TRUE) {
  relu <- function(x) pmax(0, x)
  sigmoid <- function(x) 1 / (1 + exp(-pmax(pmin(x, 10), -10)))
  
  gcn_layer <- function(A, X, W) {
    D <- diag(rowSums(A) + 1e-5)
    D_inv_sqrt <- diag(1 / sqrt(diag(D)))
    A_hat <- D_inv_sqrt %*% A %*% D_inv_sqrt
    A_hat %*% X %*% W
  }
  
  encoder <- function(A, X, W_mu, W, W_0) {
    H <- gcn_layer(A, X, W)
    mu <- H %*% W_mu
    log_var <- H %*% W_0
    list(mu = mu, log_var = log_var)
  }
  
  reparameterize <- function(mu, log_var) {
    epsilon <- matrix(rnorm(length(mu)), nrow = nrow(mu))
    log_var <- pmax(pmin(log_var, 10), -10)
    mu + exp(0.5 * log_var) * epsilon
  }
  
  decoder <- function(Z) {
    sigmoid(Z %*% t(Z))
  }
  
  log_likelihood <- function(A, Z) {
    A_pred <- decoder(Z)
    A_pred <- pmax(pmin(A_pred, 1 - 1e-10), 1e-10)
    sum(A * log(A_pred) + (1 - A) * log(1 - A_pred))
  }
  
  kl_divergence <- function(mu, log_var) {
    -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
  }
  
  elbo <- function(A, X, W_mu, W, W_0) {
    enc <- encoder(A, X, W_mu, W, W_0)
    Z <- reparameterize(enc$mu, enc$log_var)
    log_likelihood(A, Z) - kl_divergence(enc$mu, enc$log_var)
  }
  
  numerical_gradient <- function(f, W) {
    epsilon <- 1e-5
    grad <- matrix(0, nrow = nrow(W), ncol = ncol(W))
    for (i in 1:nrow(W)) {
      for (j in 1:ncol(W)) {
        W_pos <- W
        W_neg <- W
        W_pos[i, j] <- W_pos[i, j] + epsilon
        W_neg[i, j] <- W_neg[i, j] - epsilon
        grad[i, j] <- (f(W_pos) - f(W_neg)) / (2 * epsilon)
      }
    }
    grad
  }
  
  d <- ncol(X)
  W_mu <- matrix(runif(d * d, min = -1, max = 1), nrow = d)
  W <- matrix(runif(d * d, min = -1, max = 1), nrow = d)
  W_0 <- matrix(runif(d * d, min = -1, max = 1), nrow = d)
  
  for (epoch in 1:num_epochs) {
    elbo_val <- elbo(A, X, W_mu, W, W_0)
    
    grad_W_mu <- numerical_gradient(function(W_mu_) elbo(A, X, W_mu_, W, W_0), W_mu)
    grad_W <- numerical_gradient(function(W_) elbo(A, X, W_mu, W_, W_0), W)
    grad_W_0 <- numerical_gradient(function(W_0_) elbo(A, X, W_mu, W, W_0_), W_0)
    
    W_mu <- W_mu + learning_rate * grad_W_mu
    W <- W + learning_rate * grad_W
    W_0 <- W_0 + learning_rate * grad_W_0
    
    if (verbose) {
      cat("Epoch:", epoch, "| ELBO:", elbo_val, "\n")
    }
  }
  
  enc <- encoder(A, X, W_mu, W, W_0)
  Z <- reparameterize(enc$mu, enc$log_var)
  A_pred <- decoder(Z)
  
  list(
    embeddings = Z,
    A_pred = A_pred,
    mu = enc$mu,
    log_var = enc$log_var,
    weights = list(W_mu = W_mu, W = W, W_0 = W_0)
  )
}
