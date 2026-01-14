import torch
import time
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
from torch.nn import functional as F
from .utils import *

class Decisive_Edge(torch.nn.Module):
    def __init__(self, out_channels, base_model, x, y, adj, val_idx, dropout=0.5):
        """
        Initializes the Edge_Weight module. This module calculates edge weights using a base model and an MLP extractor.

        Args:
            out_channels (int): The number of output channels for the base model.
            base_model (torch.nn.Module): The base model used to generate node embeddings (actually the logits).
            dropout (float): The dropout probability for the MLP extractor.

        Attributes:
            base_model (torch.nn.Module): The base model used to generate node embeddings (actually the logits).
            extractor (nn.MLP): The multi-layer perceptron used to extract edge weights from node embeddings.

        Note:
            The parameters of the base_model are set to not require gradients during training.
        """
        super(Decisive_Edge, self).__init__()
        
        self.device = next(base_model.parameters()).device if len(list(base_model.parameters())) > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = base_model
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.adj = adj
        self.val_idx = val_idx

        # Input to the MLP is a concatenation of two node embeddings, each of size out_channels.
        # Create a custom MLP since torch_geometric.nn.MLP may not be available
        self.extractor = torch.nn.Sequential(
            torch.nn.Linear(out_channels*2, out_channels*4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_channels*4, out_channels*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_channels*2, 1)
        ).to(self.device)

        for para in self.base_model.parameters():
            para.requires_grad = False

        self.weight_train()

    def forward(self, x, adj, **kwargs):
        """
        Forward pass through the Edge_Weight module.
        Args:
            x (torch.Tensor): Input node features.
            adj (torch.Tensor): Adjacency matrix of the graph.

        Returns:
            torch.Tensor: The output logits after applying the base model with the computed edge weights.
        """
        x, adj = x.to(self.device), adj.to(self.device)
        decisive_adj = self.get_weight(x, adj)
        logits = self.base_model(x, decisive_adj)
        return logits

    def get_weight(self, x, adj):
        num_nodes = x.shape[0]
        emb = self.base_model(x, adj)
        edge_index, _ = dense_to_sparse(adj)
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        f12 = torch.cat([f1, f2], dim=-1)
        edge_weight = self.extractor(f12).reshape(-1)
        adj = torch.sparse_coo_tensor(
                    edge_index.to(self.device),
                    edge_weight,
                    size=(num_nodes, num_nodes)
                ).to_dense().relu()
        return adj

    def weight_train(self, patience=10):
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
        optimizer = torch.optim.Adam(self.extractor.parameters(), lr=0.01, weight_decay=5e-4)
        for epoch in range(250):
            self.train()
            optimizer.zero_grad()
            output = self(self.x, self.adj)
            loss = F.cross_entropy(output[self.val_idx], self.y[self.val_idx])
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

class DCGC(torch.nn.Module):
    def __init__(self, base_model, features, labels, adj, val_mask, dropout=0.5, alpha=0.5, beta=10):
        super(DCGC, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Determine output channels from base model
        with torch.no_grad():
            base_model.eval()
            sample_out = base_model(features[:10].to(self.device), adj[:10, :10].to(self.device))
            out_channels = sample_out.shape[1]

        self.model = Decisive_Edge(out_channels, base_model, features, labels, adj, val_mask, dropout)
        self.x = features.to(self.device)
        self.y = labels.to(self.device)
        self.adj = adj.to(self.device)
        self.val_idx = val_mask.to(self.device)
        self.alpha = alpha
        self.beta = beta

        for para in self.model.parameters():
            para.requires_grad = False

    def forward(self, x, adj, **kwargs):
        decisive_weight = self.model.get_weight(x, adj)
        homo_weight = self.get_homo_weight(x, adj, self.alpha, self.beta)

        unified_weight = decisive_weight * homo_weight
        logits = self.model.base_model(x, unified_weight)

        return logits
                                                 
    def get_homo_weight(self, x, adj, alpha=0.5, beta=10):
        num_nodes = x.shape[0]
        edge_index, _ = dense_to_sparse(adj)
        with torch.no_grad():
            self.model.eval()
            output = self.model(x, adj)
            pred = F.softmax(output, dim=1)

        pred = torch.exp(beta * pred)
        pred /= torch.sum(pred, dim=1, keepdim=True)

        col, row = edge_index
        coefficient = torch.norm(pred[col] - pred[row], dim=1)
        coefficient = 1 / (coefficient + alpha)

        homo_weight = torch.sparse_coo_tensor(
                                    edge_index.to(self.device),
                                    coefficient.reshape(-1),
                                    size=(num_nodes, num_nodes)
                                    ).to_dense()
                                
        return homo_weight