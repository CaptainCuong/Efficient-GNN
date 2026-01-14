"""
GATS (Graph Attention Temperature Scaling) for CompatibleGCN

A CompatibleGCN-compatible implementation of GATS calibration method.
GATS learns node-specific temperatures using graph attention mechanisms
and spatial information to improve calibration.

Reference: Based on the original GATS implementation adapted for CompatibleGCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Optional, Tuple
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree, dense_to_sparse

from .utils import accuracy


def shortest_path_length(edge_index, mask, max_hop, device):
    """
    Return the shortest path length to the mask for every node
    """
    dist_to_train = torch.ones_like(mask, dtype=torch.long, device=device) * torch.iinfo(torch.long).max
    seen_mask = torch.clone(mask).to(device)

    for hop in range(max_hop):
        current_hop = torch.nonzero(mask, as_tuple=False).squeeze(-1).to(device)
        dist_to_train[mask] = hop
        next_hop = torch.zeros_like(mask, dtype=torch.bool, device=device)

        for node in current_hop:
            node_mask = edge_index[0, :] == node
            nbrs = edge_index[1, node_mask]
            next_hop[nbrs] = True

        # mask for the next hop shouldn't be seen before
        mask = torch.logical_and(next_hop, ~seen_mask)
        seen_mask[next_hop] = True

        if not mask.any():  # No more nodes to process
            break

    return dist_to_train


class CalibAttentionLayer(MessagePassing):
    """
    Calibration attention layer that learns node-specific temperatures
    using graph attention and spatial information.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_index: torch.Tensor,
        num_nodes: int,
        train_mask: torch.Tensor,
        dist_to_train: Optional[torch.Tensor] = None,
        heads: int = 8,
        negative_slope: float = 0.2,
        bias: float = 1.0,
        self_loops: bool = True,
        fill_value: str = 'mean',
        bfs_depth: int = 2,
        device: str = 'cpu',
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.fill_value = fill_value
        self.edge_index = edge_index.to(device)
        self.num_nodes = num_nodes

        self.temp_lin = Linear(in_channels, heads, bias=False, weight_initializer='glorot')

        # Learnable parameters
        self.conf_coef = Parameter(torch.zeros([]))
        self.bias = Parameter(torch.ones(1) * bias)
        self.train_a = Parameter(torch.ones(1))
        self.dist1_a = Parameter(torch.ones(1))

        # Compute distances to nearest training node
        if dist_to_train is None:
            dist_to_train = shortest_path_length(edge_index, train_mask, bfs_depth, device)

        self.register_buffer('dist_to_train', dist_to_train)

        self.reset_parameters()

        if self_loops:
            self.edge_index, _ = remove_self_loops(self.edge_index, None)
            self.edge_index, _ = add_self_loops(
                self.edge_index, None, fill_value=self.fill_value, num_nodes=num_nodes
            )

    def reset_parameters(self):
        self.temp_lin.reset_parameters()

    def forward(self, x: torch.Tensor):
        N, H = self.num_nodes, self.heads

        # Normalize input features
        normalized_x = x - torch.min(x, 1, keepdim=True)[0]
        normalized_x /= (torch.max(x, 1, keepdim=True)[0] - torch.min(x, 1, keepdim=True)[0] + 1e-8)

        # Get sorted features and compute temperature deltas
        x_sorted = torch.sort(normalized_x, -1)[0]
        temp = self.temp_lin(x_sorted)

        # Assign spatial coefficients
        a_cluster = torch.ones(N, dtype=torch.float32, device=x.device)
        a_cluster[self.dist_to_train == 0] = self.train_a
        a_cluster[self.dist_to_train == 1] = self.dist1_a

        # Confidence for smoothing
        conf = F.softmax(x, dim=1).amax(-1)
        deg = degree(self.edge_index[0, :], self.num_nodes)
        deg_inverse = 1 / deg
        deg_inverse[deg_inverse == float('inf')] = 0

        # Message passing
        out = self.propagate(
            self.edge_index,
            temp=temp.view(N, H) * a_cluster.unsqueeze(-1),
            alpha=x / a_cluster.unsqueeze(-1),
            conf=conf
        )

        sim, dconf = out[:, :-1], out[:, -1:]
        out = F.softplus(sim + self.conf_coef * dconf * deg_inverse.unsqueeze(-1))
        out = out.mean(dim=1) + self.bias

        return out.unsqueeze(1)

    def message(
        self,
        temp_j: torch.Tensor,
        alpha_j: torch.Tensor,
        alpha_i: torch.Tensor,
        conf_i: torch.Tensor,
        conf_j: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int]
    ) -> torch.Tensor:

        alpha = (alpha_j * alpha_i).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)

        # Agreement smoothing + Confidence smoothing
        return torch.cat([
            (temp_j * alpha.unsqueeze(-1).expand_as(temp_j)),
            (conf_i - conf_j).unsqueeze(-1)
        ], -1)


class GATSCalibrator(nn.Module):
    """
    GATS Calibrator compatible with CompatibleGCN.

    This version follows the same interface pattern as other calibration methods
    (TemperatureScaling, VectorScaling, MatrixScaling) for seamless integration.
    """

    def __init__(
        self,
        base_model,
        features,
        labels,
        adj,
        val_mask,
        heads: int = 8,
        bias: float = 1.0,
        bfs_depth: int = 2
    ):
        super(GATSCalibrator, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = base_model.to(self.device)
        self.x = features.to(self.device)
        self.y = labels.to(self.device)
        self.adj = adj.to(self.device)
        self.val_idx = val_mask.to(self.device)

        # Generate edge_index from adjacency matrix
        self.edge_index, _ = dense_to_sparse(self.adj)

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get number of classes from base model output
        with torch.no_grad():
            sample_output = self.base_model(self.x, self.adj)
            num_classes = sample_output.shape[1]

        # Create train mask for calibration attention
        train_mask = torch.zeros(self.x.size(0), dtype=torch.bool, device=self.device)
        train_mask[self.val_idx] = True

        # Initialize calibration attention layer
        self.calib_attention = CalibAttentionLayer(
            in_channels=num_classes,
            out_channels=1,
            edge_index=self.edge_index,
            num_nodes=self.x.size(0),
            train_mask=train_mask,
            heads=heads,
            bias=bias,
            bfs_depth=bfs_depth,
            device=self.device
        ).to(self.device)

        self.calib_train()

    def forward(self, x, adj, **kwargs):
        """Forward pass with temperature scaling."""
        logits = self.base_model(x, adj)
        temperature = self.graph_temperature_scale(logits)
        return F.log_softmax(logits / temperature, dim=1)

    def graph_temperature_scale(self, logits):
        """Apply graph-based temperature scaling."""
        temperature = self.calib_attention(logits).view(self.x.size(0), -1)
        return temperature.expand(self.x.size(0), logits.size(1))

    def calib_train(self, patience=10, epochs=250, lr=0.01, weight_decay=5e-4):
        """Train the calibration model."""
        t = time.time()
        best_loss = float('inf')
        patience_counter = patience

        optimizer = torch.optim.Adam(
            self.calib_attention.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        for epoch in range(epochs):
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


def calibrate_with_gats(base_model, features, labels, adj, val_mask, **kwargs):
    """
    Convenience function to apply GATS calibration.

    Args:
        base_model: The trained base model to calibrate
        features: Node features
        labels: Node labels
        adj: Adjacency matrix
        val_mask: Validation set mask for calibration
        **kwargs: Additional parameters for GATS

    Returns:
        GATSCalibrator: The calibrated model
    """
    return GATSCalibrator(base_model, features, labels, adj, val_mask, **kwargs)


# For backward compatibility
GATS = GATSCalibrator