from torch import nn
import torch
import time
from torch.nn import functional as F
from .utils import accuracy

class MatrixScaling(nn.Module):
    '''
    Implement matrix scaling for model calibration. 
    Adjust the logits using a learned scaling matrix
    '''
    def __init__(self, base_model, x, y, adj, val_idx):
        """
        Initialize the MatrixScaling class for model calibration.

        Parameters:
        - model (nn.Module): The base model to be calibrated.
        - nfeat (int): The number of input features.
        - n_bins (int): The number of bins for ECE calculation.

        The class initializes the scaling matrix (W) and bias (b) as trainable parameters.
        It also sets the regularization parameters Lambda and mu, and the maximum number of iterations for optimization.
        """
        super(MatrixScaling, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = base_model
        self.n_classes = y.max().item() + 1  # Calculate n_classes from labels
        self.W = nn.Parameter(torch.ones(self.n_classes, self.n_classes))  # Initialize W as a diagonal matrix with ones
        self.b = nn.Parameter(torch.ones(self.n_classes))  # Initialize b as a vector with ones
        self.Lambda = 1
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.adj = adj.to(self.device)
        self.val_idx = val_idx.to(self.device)
        
        # Freeze the parameters of the base model
        for para in self.model.parameters():
            para.requires_grad = False

        self.to(self.device)
        self.calib_train()

    def forward(self, features, adj, **kwargs):
        logits = self.model(features, adj)
        last_class_logits = logits[:, -1].unsqueeze(1)  # Shape: [2995, 1]
        logits = logits - last_class_logits
        return self.matrix_scale(logits)

    def matrix_scale(self, logits):
        """
        Perform matrix scaling on logits
        """
        logits = logits.to(self.device)
        # Expand temperature to match the size of logits
        logits = torch.matmul(logits, self.W)
        logits = logits + self.b
        return logits
    
    def calib_train(self, patience=10):
        t = time.time()
        best_loss = float('inf')
        patience_counter = patience
        optimizer = torch.optim.Adam([self.W, self.b], lr=0.01, weight_decay=5e-4)
        for epoch in range(250):
            self.train()
            optimizer.zero_grad()
            output = self(self.x, self.adj)
            loss = F.nll_loss(output[self.val_idx], self.y[self.val_idx]) +\
                   self.Lambda * torch.sum(torch.abs(self.W - torch.eye(self.n_classes).to(self.device)))
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


def calibrate_with_matrix_scaling(base_model, x, y, adj, val_idx):
    """
    Convenience function to apply matrix scaling calibration.

    Args:
        base_model: The trained base model to calibrate
        x: Node features
        y: Node labels
        adj: Adjacency matrix
        val_idx: Validation set indices for calibration

    Returns:
        MatrixScaling: The calibrated model
    """
    return MatrixScaling(base_model, x, y, adj, val_idx)