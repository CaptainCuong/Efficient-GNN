# WATS Implementation (Post-hoc Calibration for GNNs)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from .utils import accuracy
from scipy.sparse import csgraph, identity, csr_matrix

'''
WATS (Wavelet-Aware Temperature Scaling) is a post-hoc calibration method designed for Graph Neural Networks (GNNs).

It leverages graph wavelet transforms to capture localized structural and spectral information around each node.

WATS uses these wavelet-based features to train a neural temperature predictor,
which outputs a personalized temperature for each node.

This node-wise temperature is then used to scale the model’s logits,
improving the alignment between predicted confidence and actual accuracy.

WATS is particularly effective in addressing calibration errors caused by graph heterogeneity.
'''

def compute_normalized_laplacian(adj):
    """Return the symmetric normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}"""
    laplacian = csgraph.laplacian(adj, normed=True)
    return laplacian

def chebyshev_polynomials(L, k, X0):
    """Compute Chebyshev polynomials T_k(L) X0 up to order k"""
    N = L.shape[0]
    T_k = [X0]
    if k > 0:
        T_k.append(L @ X0)
    for i in range(2, k + 1):
        T_k.append(2 * L @ T_k[-1] - T_k[-2])
    return T_k

def graph_wavelet_features(adj_matrix, k=3, s=0.8):
    """
    Compute graph wavelet features using heat kernel and Chebyshev approximation.
    
    Parameters:
        adj_matrix: (N, N) numpy or sparse matrix, the adjacency matrix.
        k: int, Chebyshev order (controls neighborhood reach).
        s: float, heat kernel diffusion scale.

    Returns:
        features: (N, k+1) array of wavelet features per node.
    """
    N = adj_matrix.shape[0]
    # Compute normalized Laplacian
    L = compute_normalized_laplacian(adj_matrix)
    # Rescale Laplacian to [-1, 1]
    L_rescaled = (2 / 2.0) * L - identity(N)

    # Input signal: log(degree + 1)
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    X0 = np.log1p(degrees).reshape(-1, 1)

    # Compute Chebyshev polynomials T_k(L) X0
    T_k = chebyshev_polynomials(L_rescaled, k, X0)

    # Heat kernel coefficients: alpha_k = exp(-s * k)
    alpha = [np.exp(-s * i) for i in range(k + 1)]

    # Combine into wavelet features
    S = sum(alpha[i] * T_k[i] for i in range(k + 1))

    # Row-wise L1 normalization
    row_sums = np.linalg.norm(S, ord=1, axis=1, keepdims=True) + 1e-8
    H = S / row_sums

    return H  # shape: (N, 1), can be extended if input X0 has more features
    
class WATS(torch.nn.Module):
    def __init__(self, base_model, features, labels, adj, val_mask):
        """
        Initializes the WATS model.

        Parameters:
        - base_model (torch.nn.Module): The base model used to compute logits.
        - features (torch.Tensor): Input node features.
        - labels (torch.Tensor): True labels for the nodes.
        - adj (torch.Tensor): Adjacency matrix of the graph.
        - val_mask (torch.Tensor): Validation set mask for calibration.
        """
        super(WATS, self).__init__()

        # Auto-detect device from base model or features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = base_model.to(self.device)
        self.x = features.to(self.device)
        self.y = labels.to(self.device)
        self.adj = adj.to(self.device)
        self.val_idx = val_mask.to(self.device)

        # Compute wavelet features
        self.wavelet_feats = graph_wavelet_features(csr_matrix(adj.cpu().numpy()), k=3, s=0.8)
        self.wavelet_feats = torch.tensor(self.wavelet_feats, dtype=torch.float32).to(self.device)
        self.net = nn.Sequential(
            nn.Linear(self.wavelet_feats.shape[1], 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(self.device)

        for para in self.net.parameters():
            para.requires_grad = True

        self.calib_train()

    def forward(self, x, adj):
        """
        Forward pass through the WATS model.
        Args:
            x (torch.Tensor): Input node features.
            adj (torch.Tensor): Adjacency matrix of the graph.

        Returns:
            torch.Tensor: The calibrated output logits.
        """
        x, adj = x.to(self.device), adj.to(self.device)
        wavelet_features = self.wavelet_feats.to(x.device)
        temperatures = self.net(wavelet_features).squeeze()
        temperatures = torch.log(torch.exp(temperatures) + torch.tensor(1.1, device=self.device)).to(self.device)

        # CompatibleGCN doesn't accept normalize parameter - it handles normalization internally
        logits = self.base_model(x, adj)
        calibrated_logits = logits / temperatures.unsqueeze(1)
        return F.log_softmax(calibrated_logits, dim=1)
    
    def calib_train(self, patience=10):
        """
        Trains the calibration model.

        This method trains the calibration model by performing a forward pass through the model and computing the loss.
        The loss is then backpropagated through the model to update the model parameters.

        Returns:
        None
        """
        t = time.time()
        best_loss = float('inf')
        patience_counter = patience
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01, weight_decay=5e-4)
        for epoch in range(250):
            self.train()
            optimizer.zero_grad()
            output = self(self.x, self.adj)
            loss = F.nll_loss(output[self.val_idx], self.y[self.val_idx])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.eval()
                acc = accuracy(output[self.val_idx], self.y[self.val_idx])
                print(f'epoch: {epoch}',
                    f'loss_calibration: {loss.item():.4f}',
                    f'acc_calibration: {acc:.4f}',
                    f'time: {time.time() - t:.4f}s')
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = patience
            else:
                patience_counter -= 1

            if patience_counter <= 0:
                print(f'Early stopping at epoch {epoch}, best loss: {best_loss:.4f}')
                break