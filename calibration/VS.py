import torch
from torch.nn import functional as F
import time
from .utils import accuracy
class VectorScaling(torch.nn.Module):
    def __init__(self, base_model, x, y, adj, val_idx):
        """
        Initialize the VectorScaling class.

        This class implements vector scaling for model calibration. It initializes
        the temperature parameter, sets up the base model, and prepares the data
        for calibration training.

        Parameters:
        -----------
        base_model : torch.nn.Module
            The pre-trained base model to be calibrated.
        x : torch.Tensor
            The input features for the model.
        y : torch.Tensor
            The ground truth labels.
        adj : torch.Tensor
            The adjacency matrix representing the graph structure.
        val_idx : torch.Tensor
            The indices of the validation set.

        Returns:
        --------
        None
        """
        super(VectorScaling, self).__init__()
        n_classes = y.max().item() + 1  # Calculate n_classes from labels
        self.temperature = torch.nn.Parameter(torch.ones(n_classes) * 1.0)  # Initialize T to 1.0
        self.base_model = base_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.adj = adj
        self.val_idx = val_idx
        for para in self.base_model.parameters():
            para.requires_grad = False

        self.calib_train()

    def forward(self, x, adj, **kwargs):
        logits = self.base_model(x.to(self.device), adj.to(self.device))
        # Apply temperature scaling
        t = torch.log(torch.exp(self.temperature) + torch.tensor(1.1)).to(self.device)
        return F.log_softmax(logits * t, dim=1)
    
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
        optimizer = torch.optim.Adam([self.temperature], lr=0.01, weight_decay=5e-4)
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


def calibrate_with_vector_scaling(base_model, x, y, adj, val_idx):
    """
    Convenience function to apply vector scaling calibration.

    Args:
        base_model: The trained base model to calibrate
        x: Node features
        y: Node labels
        adj: Adjacency matrix
        val_idx: Validation set indices for calibration

    Returns:
        VectorScaling: The calibrated model
    """
    return VectorScaling(base_model, x, y, adj, val_idx)