"""
GETS (Graph Expert Temperature Scaling) for CompatibleGCN

A CompatibleGCN-compatible implementation of GETS calibration method.
GETS uses a mixture of experts approach with multiple graph neural networks
to learn node-specific temperature scaling parameters.

Reference: Adapted from the original GETS implementation for CompatibleGCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Optional
from torch.distributions.normal import Normal
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.utils import degree

from .utils import accuracy


class GCN_Expert(nn.Module):
    """GCN Expert for GETS mixture of experts."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        dropout_rate: float,
        num_layers: int,
        device: str,
        expert_config: List[str],
        feature_dim: int,
        feature_hidden_dim: int,
        degree_hidden_dim: int
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.expert_config = expert_config
        self.device = device

        # Calculate input channels based on configuration
        in_channels = 0
        if "logits" in expert_config:
            in_channels += num_classes
        if "features" in expert_config:
            self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)
            in_channels += feature_hidden_dim
        if "degrees" in expert_config:
            in_channels += degree_hidden_dim

        # Build layer architecture
        feature_list = [in_channels]
        for _ in range(num_layers - 2):
            feature_list.append(hidden_dim)
        feature_list.append(num_classes)

        self.convs = nn.ModuleList()
        for i in range(len(feature_list) - 1):
            self.convs.append(GCNConv(feature_list[i], feature_list[i + 1]))

        self.degree_dim = degree_hidden_dim

    def forward(self, logits, features, edge_index):
        inputs = []
        if "logits" in self.expert_config:
            inputs.append(logits)
        if "features" in self.expert_config:
            features_proj = self.proj_feature(features)
            inputs.append(features_proj)
        if "degrees" in self.expert_config:
            if not hasattr(self, "degrees"):
                # Calculate degrees from edge_index
                num_nodes = logits.size(0)
                deg = degree(edge_index[0], num_nodes) + degree(edge_index[1], num_nodes)
                max_degree = int(deg.max().item()) + 1
                self.degree_embedder = nn.Embedding(max_degree, self.degree_dim).to(self.device)
                self.degrees = deg.long()
            degree_embeds = self.degree_embedder(self.degrees)
            inputs.append(degree_embeds)

        x = torch.cat(inputs, dim=-1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, training=self.training)

        return x


class GAT_Expert(nn.Module):
    """GAT Expert for GETS mixture of experts."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        dropout_rate: float,
        num_layers: int,
        device: str,
        expert_config: List[str],
        feature_dim: int,
        feature_hidden_dim: int,
        degree_hidden_dim: int,
        num_heads: int = 2
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.expert_config = expert_config
        self.device = device
        self.num_heads = num_heads

        # Calculate input channels
        in_channels = 0
        if "logits" in expert_config:
            in_channels += num_classes
        if "features" in expert_config:
            self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)
            in_channels += feature_hidden_dim
        if "degrees" in expert_config:
            in_channels += degree_hidden_dim

        # Build GAT layers
        feature_list = [in_channels] + [hidden_dim] * (num_layers - 1)

        self.convs = nn.ModuleList()
        for i in range(len(feature_list) - 1):
            out_dim = feature_list[i + 1] // num_heads
            self.convs.append(GATConv(feature_list[i], out_dim, heads=num_heads, dropout=dropout_rate))

        self.final_proj = nn.Linear(hidden_dim, num_classes)
        self.degree_dim = degree_hidden_dim

    def forward(self, logits, features, edge_index):
        inputs = []
        if "logits" in self.expert_config:
            inputs.append(logits)
        if "features" in self.expert_config:
            features_proj = self.proj_feature(features)
            inputs.append(features_proj)
        if "degrees" in self.expert_config:
            if not hasattr(self, "degrees"):
                num_nodes = logits.size(0)
                deg = degree(edge_index[0], num_nodes) + degree(edge_index[1], num_nodes)
                max_degree = int(deg.max().item()) + 1
                self.degree_embedder = nn.Embedding(max_degree, self.degree_dim).to(self.device)
                self.degrees = deg.long()
            degree_embeds = self.degree_embedder(self.degrees)
            inputs.append(degree_embeds)

        x = torch.cat(inputs, dim=-1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.final_proj(x)
        return x


class GIN_Expert(nn.Module):
    """GIN Expert for GETS mixture of experts."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        dropout_rate: float,
        num_layers: int,
        device: str,
        expert_config: List[str],
        feature_dim: int,
        feature_hidden_dim: int,
        degree_hidden_dim: int
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.expert_config = expert_config
        self.device = device

        # Calculate input channels
        in_channels = 0
        if "logits" in expert_config:
            in_channels += num_classes
        if "features" in expert_config:
            self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)
            in_channels += feature_hidden_dim
        if "degrees" in expert_config:
            in_channels += degree_hidden_dim

        # Build GIN layers
        feature_list = [in_channels]
        for _ in range(num_layers - 2):
            feature_list.append(hidden_dim)
        feature_list.append(num_classes)

        self.convs = nn.ModuleList()
        for i in range(len(feature_list) - 1):
            mlp = nn.Sequential(
                nn.Linear(feature_list[i], feature_list[i + 1]),
                nn.ReLU(),
                nn.Linear(feature_list[i + 1], feature_list[i + 1])
            )
            self.convs.append(GINConv(mlp))

        self.degree_dim = degree_hidden_dim

    def forward(self, logits, features, edge_index):
        inputs = []
        if "logits" in self.expert_config:
            inputs.append(logits)
        if "features" in self.expert_config:
            features_proj = self.proj_feature(features)
            inputs.append(features_proj)
        if "degrees" in self.expert_config:
            if not hasattr(self, "degrees"):
                num_nodes = logits.size(0)
                deg = degree(edge_index[0], num_nodes) + degree(edge_index[1], num_nodes)
                max_degree = int(deg.max().item()) + 1
                self.degree_embedder = nn.Embedding(max_degree, self.degree_dim).to(self.device)
                self.degrees = deg.long()
            degree_embeds = self.degree_embedder(self.degrees)
            inputs.append(degree_embeds)

        x = torch.cat(inputs, dim=-1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, training=self.training)

        return x


class GETSCalibrator(nn.Module):
    """
    GETS Calibrator compatible with CompatibleGCN.

    Graph Expert Temperature Scaling using mixture of experts approach
    with multiple GNN architectures for node-specific temperature learning.
    """

    def __init__(
        self,
        base_model,
        num_classes,
        device,
        conf,
        x=None,
        y=None,
        adj=None,
        val_idx=None,
        num_experts: int = 3,
        expert_select: int = 2,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1,
        num_layers: int = 2,
        feature_hidden_dim: int = 32,
        degree_hidden_dim: int = 16,
        noisy_gating: bool = True,
        loss_coef: float = 1e-2,
        backbone: str = 'gcn'
    ):
        super().__init__()

        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = base_model.to(self.device)
        self.conf = conf
        self.num_classes = num_classes

        # Store data for training - will be set in fit method
        self.x = None
        self.y = None
        self.adj = None
        self.val_idx = None
        self.edge_index = None

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Model dimensions - will be set in fit method
        self.feature_dim = None

        # GETS parameters
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = expert_select  # Number of experts to select
        self.loss_coef = loss_coef
        self.backbone = backbone

        # Expert configurations - different combinations of inputs
        expert_configs = [
            ["logits", "features"],
            ["logits", "degrees"],
            ["features", "degrees"],
            ["logits", "features", "degrees"]
        ]
        # Use only the first num_experts configurations
        expert_configs = expert_configs[:num_experts]

        # Store hyperparameters - will build networks in fit method
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.feature_hidden_dim = feature_hidden_dim
        self.degree_hidden_dim = degree_hidden_dim
        self.backbone = backbone
        self.expert_configs = [
            ["logits", "features"],
            ["logits", "degrees"],
            ["features", "degrees"],
            ["logits", "features", "degrees"]
        ][:num_experts]

        # Will be initialized in fit method
        self.experts = None
        self.proj_feature = None
        self.w_gate = None
        self.w_noise = None
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        assert self.k <= self.num_experts

    def cv_squared(self, x):
        """Squared coefficient of variation for load balancing."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute load per expert."""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper for noisy top-k gating."""
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating mechanism."""
        clean_logits = x @ self.w_gate

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # Get top-k experts
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def forward(self, x, adj, **kwargs):
        """Forward pass with expert temperature scaling."""
        logits = self.base_model(x, adj)

        # Get gating input
        features_trans = self.proj_feature(x)
        gating_input = torch.cat([features_trans, logits], dim=1)

        # Compute expert gates
        node_gates, load = self.noisy_top_k_gating(gating_input, self.training)
        importance = node_gates.sum(0)

        # Expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.experts[i](logits, x, self.edge_index)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # Combine expert outputs with gates
        temperature = (expert_outputs * node_gates.unsqueeze(-1)).sum(dim=1)
        calibrated_logits = logits * F.softplus(temperature)

        # Store auxiliary loss for training
        self.aux_loss = self.cv_squared(importance) + self.cv_squared(load)
        self.aux_loss *= self.loss_coef

        return F.log_softmax(calibrated_logits, dim=1)

    def fit(self, adj, features, labels, masks):
        """Initialize networks and train the calibration model."""
        # Store data
        self.x = features.to(self.device)
        self.y = labels.to(self.device)
        self.adj = adj.to(self.device)
        self.val_idx = masks[1].to(self.device)  # validation mask
        self.feature_dim = features.shape[1]

        # Convert adjacency matrix to edge_index
        adj_indices = torch.nonzero(self.adj, as_tuple=False)
        self.edge_index = adj_indices.t().contiguous().to(self.device)

        # Build networks now that we have the data
        self._build_networks()

        # Train the model
        self.calib_train()
        return self

    def _build_networks(self):
        """Build expert networks and gating mechanism."""
        # Create experts based on backbone
        self.experts = nn.ModuleList()
        for i in range(self.num_experts):
            if self.backbone == 'gcn':
                expert = GCN_Expert(
                    num_classes=self.num_classes,
                    hidden_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate,
                    num_layers=self.num_layers,
                    device=self.device,
                    expert_config=self.expert_configs[i],
                    feature_dim=self.feature_dim,
                    feature_hidden_dim=self.feature_hidden_dim,
                    degree_hidden_dim=self.degree_hidden_dim
                ).to(self.device)
            elif self.backbone == 'gat':
                expert = GAT_Expert(
                    num_classes=self.num_classes,
                    hidden_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate,
                    num_layers=self.num_layers,
                    device=self.device,
                    expert_config=self.expert_configs[i],
                    feature_dim=self.feature_dim,
                    feature_hidden_dim=self.feature_hidden_dim,
                    degree_hidden_dim=self.degree_hidden_dim
                ).to(self.device)
            elif self.backbone == 'gin':
                expert = GIN_Expert(
                    num_classes=self.num_classes,
                    hidden_dim=self.hidden_dim,
                    dropout_rate=self.dropout_rate,
                    num_layers=self.num_layers,
                    device=self.device,
                    expert_config=self.expert_configs[i],
                    feature_dim=self.feature_dim,
                    feature_hidden_dim=self.feature_hidden_dim,
                    degree_hidden_dim=self.degree_hidden_dim
                ).to(self.device)
            else:
                raise NotImplementedError(f"Backbone {self.backbone} not implemented")
            self.experts.append(expert)

        # Gating network
        self.proj_feature = nn.Linear(self.feature_dim, self.feature_hidden_dim).to(self.device)
        gate_input_dim = self.feature_hidden_dim + self.num_classes
        self.w_gate = nn.Parameter(torch.zeros(gate_input_dim, self.num_experts, device=self.device), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(gate_input_dim, self.num_experts, device=self.device), requires_grad=True)

        # Initialize parameters
        nn.init.normal_(self.w_gate, mean=0.0, std=0.02)
        nn.init.normal_(self.w_noise, mean=0.0, std=0.02)

        self.register_buffer("mean", torch.tensor([0.0], device=self.device))
        self.register_buffer("std", torch.tensor([1.0], device=self.device))

    def calib_train(self, patience=10, epochs=250, lr=0.01, weight_decay=5e-4):
        """Train the calibration model."""
        t = time.time()
        best_loss = float('inf')
        patience_counter = patience

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            output = self(self.x, self.adj)
            loss = F.nll_loss(output[self.val_idx], self.y[self.val_idx])

            # Add auxiliary loss for expert balancing
            if hasattr(self, 'aux_loss'):
                loss += self.aux_loss

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


def calibrate_with_gets(base_model, x, y, adj, val_idx, edge_index, **kwargs):
    """
    Convenience function to apply GETS calibration.

    Args:
        base_model: The trained base model to calibrate
        x: Node features
        y: Node labels
        adj: Adjacency matrix
        val_idx: Validation set indices for calibration
        edge_index: Edge index tensor for graph structure
        **kwargs: Additional parameters for GETS

    Returns:
        GETSCalibrator: The calibrated model
    """
    return GETSCalibrator(base_model, x, y, adj, val_idx, edge_index, **kwargs)