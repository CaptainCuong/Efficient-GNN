"""
Temperature Scaling for Model Calibration

Temperature Scaling (TS) is a simple post-hoc calibration method that learns
a single scalar parameter (temperature) to rescale the logits before applying softmax.

Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
https://arxiv.org/abs/1706.04599
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from utils.ece import calculate_average_ece
from .utils import accuracy

class TemperatureScaling(torch.nn.Module):
    def __init__(self, base_model, x, y, adj, val_idx):
        super(TemperatureScaling, self).__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.0)  # Initialize T to 1.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = base_model.to(self.device)
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.adj = adj.to(self.device)
        self.val_idx = val_idx.to(self.device)

        for para in self.base_model.parameters():
            para.requires_grad = False

        self.calib_train()

    def forward(self, x, adj, **kwargs):
        logits = self.base_model(x, adj)
        # Apply temperature scaling
        t = torch.log(torch.exp(self.temperature) + torch.tensor(1.1)).to(self.device)
        return F.log_softmax(logits * t, dim=1)
    
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


def calibrate_with_temperature_scaling(base_model, x, y, adj, val_idx):
    """
    Convenience function to apply temperature scaling calibration.

    Args:
        base_model: The trained base model to calibrate
        x: Node features
        y: Node labels
        adj: Adjacency matrix
        val_idx: Validation set indices for calibration

    Returns:
        TemperatureScaling: The calibrated model
    """
    return TemperatureScaling(base_model, x, y, adj, val_idx)


def evaluate_calibration(model, x, y, adj, test_idx):
    """
    Evaluate calibration performance of a model.

    Args:
        model: The model to evaluate
        x: Node features
        y: Node labels
        adj: Adjacency matrix
        test_idx: Test set indices

    Returns:
        dict: Dictionary containing calibration metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    with torch.no_grad():
        logits = model(x, adj)
        if hasattr(logits, 'exp'):  # log_softmax output
            probs = logits.exp()
        else:  # raw logits
            probs = F.softmax(logits, dim=1)

        test_probs = probs[test_idx]
        test_labels = y[test_idx]

        # Calculate accuracy
        pred_labels = torch.argmax(test_probs, dim=1)
        accuracy = (pred_labels == test_labels).float().mean().item()

        # Calculate ECE
        n_classes = test_probs.shape[1]  # Number of classes from probs shape
        ece = calculate_average_ece(test_probs.cpu().numpy(), test_labels.cpu().numpy(), n_classes, logits=False)

        # Calculate confidence
        confidence = torch.max(test_probs, dim=1)[0]
        avg_confidence = confidence.mean().item()

        return {
            'accuracy': accuracy,
            'ece': ece,
            'avg_confidence': avg_confidence
        }


def plot_reliability_diagram(probs, labels, n_bins=10, title="Reliability Diagram"):
    """
    Plot reliability diagram for calibration visualization.

    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins for binning
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Convert to numpy if needed
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # Get max probabilities and predictions
    max_probs = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies = []
    confidences = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
        else:
            accuracies.append(0)
            confidences.append(0)

    # Plot
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.bar(confidences, accuracies, width=0.08, alpha=0.7, label='Model')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_confidence_histogram(probs, labels, n_bins=10, title="Confidence Histogram"):
    """
    Plot confidence histogram.

    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Convert to numpy if needed
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    max_probs = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)

    # Plot histograms
    ax.hist(max_probs[correct], bins=n_bins, alpha=0.7, label='Correct', density=True)
    ax.hist(max_probs[~correct], bins=n_bins, alpha=0.7, label='Incorrect', density=True)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def comprehensive_calibration_analysis(model, x, y, adj, test_idx, save_path=None):
    """
    Perform comprehensive calibration analysis.

    Args:
        model: Model to analyze
        x: Node features
        y: Node labels
        adj: Adjacency matrix
        test_idx: Test indices
        save_path: Optional path to save plots

    Returns:
        dict: Analysis results
    """
    # Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(x, adj)
        if hasattr(logits, 'exp'):
            probs = logits.exp()
        else:
            probs = F.softmax(logits, dim=1)

    test_probs = probs[test_idx]
    test_labels = y[test_idx]

    # Calculate metrics
    results = evaluate_calibration(model, x, y, adj, test_idx)

    # Create plots
    rel_fig = plot_reliability_diagram(test_probs, test_labels)
    conf_fig = plot_confidence_histogram(test_probs, test_labels)

    if save_path:
        rel_fig.savefig(f"{save_path}_reliability.png")
        conf_fig.savefig(f"{save_path}_confidence.png")

    return {
        'metrics': results,
        'reliability_fig': rel_fig,
        'confidence_fig': conf_fig
    }


def load_calibrated_model(model_path, base_model, x, y, adj, val_idx):
    """
    Load a pre-trained calibrated model.

    Args:
        model_path: Path to saved model
        base_model: Base model architecture
        x, y, adj, val_idx: Model parameters

    Returns:
        TemperatureScaling: Loaded calibrated model
    """
    calibrated_model = TemperatureScaling(base_model, x, y, adj, val_idx)
    calibrated_model.load_state_dict(torch.load(model_path))
    return calibrated_model