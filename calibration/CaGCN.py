import torch
from torch import nn
import time
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .utils import *

def calibration_loss(output, labels):
    """
    Calculates the calibration loss for a given set of model outputs and corresponding labels.

    The calibration loss is a measure of how well the model's confidence matches its accuracy.
    It is calculated as the average difference between the model's highest confidence and second highest confidence
    for correctly classified examples, and the difference between the model's highest confidence and second highest confidence
    for incorrectly classified examples.

    Parameters:
    - output (torch.Tensor): A tensor of shape (batch_size, num_classes) containing the model's raw output logits.
    - labels (torch.Tensor): A tensor of shape (batch_size) containing the true labels for the examples.

    Returns:
    - loss (torch.Tensor): A scalar tensor representing the average calibration loss.
    """
    # Apply softmax to the output logits
    output = torch.softmax(output, dim=1)

    # Get the predicted labels and indices for correctly and incorrectly classified examples
    pred_max_index = torch.max(output, 1)[1]  # argmax
    correct_i = torch.where(pred_max_index == labels)  # correct index
    incorrect_i = torch.where(pred_max_index != labels)  # incorrect index

    # Sort the output logits in descending order
    output = torch.sort(output, dim=1, descending=True)

    # Extract the highest confidence and second highest confidence for each example
    pred, sub_pred = output[0][:, 0], output[0][:, 1]

    # Calculate the calibration loss
    loss = (torch.sum(1 - pred[correct_i] + sub_pred[correct_i]) + torch.sum(pred[incorrect_i] - sub_pred[incorrect_i])) / labels.size()[0]

    return loss


class CaGCN(nn.Module):
    def __init__(self, base_model, x, y, adj, val_idx):
        """
        Initializes the CaGCN model.

        The CaGCN model is a combination of a base model and a scaling model. The base model is used to compute logits,
        and the scaling model is used to calibrate the logits. The initialization sets up the base model, scaling model,
        input features, adjacency matrix, and validation indices. It also freezes the parameters of the base model.

        Parameters:
        - base_model (torch.nn.Module): The base model used to compute logits.
        - x (torch.Tensor): A tensor of shape (num_nodes, num_features) representing the input node features.
        - y (torch.Tensor): A tensor of shape (num_nodes) representing the true labels for the nodes.
        - adj (torch.Tensor): A tensor of shape (num_nodes, num_nodes) representing the adjacency matrix of the graph.
        - val_idx (torch.Tensor): A tensor of indices representing the validation set for model calibration.

        Returns:
        None
        """
        super(CaGCN, self).__init__()
        self.base_model = base_model
        self.n_classes = y.max().item() + 1  # Calculate n_classes from labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaling_model = [GCNConv(in_channels=self.n_classes, out_channels=self.n_classes).to(self.device),
                              GCNConv(in_channels=self.n_classes, out_channels=self.n_classes).to(self.device)
                              ]
        self.dropout = nn.Dropout(p=0.5).to(self.device)

        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.adj = adj.to(self.device)
        self.val_idx = val_idx.to(self.device)

        for para in self.base_model.parameters():
            para.requires_grad = False

        self.calib_train()

    def forward(self, x, adj, **kwargs):
        """
        Performs a forward pass through the CaGCN model.

        The forward pass computes the logits by passing the input features and adjacency matrix through the base model,
        and then scales the logits using a scaling model. The scaling is done by applying a GCNConv layer followed by a
        logarithmic transformation.

        Parameters:
        - x (torch.Tensor): A tensor of shape (num_nodes, num_features) representing the input node features.
        - adj (torch.Tensor): A tensor of shape (num_nodes, num_nodes) representing the adjacency matrix of the graph.

        Returns:
        - output (torch.Tensor): A tensor of shape (num_nodes, num_classes) representing the scaled logits for each node.
        """
        # TO DO: check if adj is a square matrix
        if adj.shape[0] != adj.shape[1] or len(adj.shape) != 2:
            raise ValueError("The adjacency matrix must be a square matrix.")

        # CompatibleGCN doesn't support normalize parameter
        logits = self.base_model(x, adj)
        edge_index, edge_weight = dense_to_sparse(adj)
        t = self.scaling_model[0](logits, edge_index.to(logits.device))
        t = torch.relu(t)
        t = self.dropout(t)
        t = self.scaling_model[1](t, edge_index.to(logits.device))
        t = torch.log(torch.exp(t) + torch.tensor(1.1).to(logits.device))
        output = logits * t

        return F.log_softmax(output, dim=1)
    
    def base_predict(self, x, adj):
        logits = self.base_model(x, adj)
        return logits
    
    def calib_train(self, patience=10, alpha=0.5):
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
        optimizer = torch.optim.Adam([_ for _ in self.scaling_model[0].parameters()] + [_ for _ in self.scaling_model[1].parameters()], lr=0.01, weight_decay=5e-4)
        for epoch in range(100):
            self.train()
            optimizer.zero_grad()
            output = self(self.x, self.adj)
            loss = F.nll_loss(output[self.val_idx], self.y[self.val_idx]) + \
                   alpha * calibration_loss(output[self.val_idx], self.y[self.val_idx])
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
