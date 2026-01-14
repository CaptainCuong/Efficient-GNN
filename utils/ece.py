import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.special import softmax
import seaborn as sns

def calculate_ece(model_outputs, labels, pos_class, logits=True, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE) for a given set of logits and labels.

    Parameters:
    model_outputs (np.ndarray): Array of shape (batch_size, num_classes) containing the predicted logits for each class.
    labels (np.ndarray): Array of shape (batch_size,) containing the true labels.
    pos_class (int): The index of the positive class for which to calculate the ECE.
    n_bins (int, optional): The number of bins to use for discretizing the predicted probabilities. Default is 10.

    Returns:
    float: The calculated Expected Calibration Error (ECE).
    """
    # Check if model outputs and labels have numpy type
    if not isinstance(model_outputs, np.ndarray) or not isinstance(labels, np.ndarray):
        raise ValueError("Input arrays must be of type numpy.ndarray.")
    
    # Check if model outputs and labels have the same number of elements
    if model_outputs.shape[0] != labels.shape[0]:
        raise ValueError("Input arrays must have the same number of elements.")
    
    # Extract the probabilities for the positive class
    if logits:
        predictions = softmax(model_outputs, axis=1)[:, pos_class]
    else:
        predictions = model_outputs[:, pos_class]
    
    # Extract positive class labels
    labels = (labels == pos_class)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges, right=True) - 1
    
    ece = 0.0

    for i in range(n_bins):
        # Get indices for the current bin
        bin_mask = bin_indices == i
        
        if np.sum(bin_mask) < 4:
            # Skip bins with less than 4 samples
            continue
        
        # Calculate accuracy and confidence in the current bin
        bin_accuracy = np.mean(labels[bin_mask])
        bin_confidence = np.mean(predictions[bin_mask])
        
        # Calculate the bin weight (proportion of total samples in the bin)
        bin_weight = np.mean(bin_mask)
        
        # Accumulate the ECE
        ece += np.abs(bin_confidence - bin_accuracy) * bin_weight
    
    return ece

def calculate_average_ece(model_outputs, labels, n_classes, logits=True, n_bins=10):
    """
    Calculate the average Expected Calibration Error (ECE) across all classes.
    
    Parameters:
    model_outputs (np.ndarray): Array of shape (batch_size, num_classes) containing 
                               the predicted logits or probabilities for each class.
    labels (np.ndarray): Array of shape (batch_size,) containing the true labels.
    n_classes (int): The number of classes in the dataset.
    logits (bool, optional): Whether the model_outputs are logits (True) or probabilities (False).
                            Default is True.
    n_bins (int, optional): The number of bins to use for discretizing the predicted probabilities.
                           Default is 10.
    
    Returns:
    float: The average Expected Calibration Error (ECE) across all classes.
    """
    ece_values = []
    
    # Calculate ECE for each class
    for class_idx in range(n_classes):
        class_ece = calculate_ece(model_outputs, labels, class_idx, logits=logits, n_bins=n_bins)
        ece_values.append(class_ece)
    
    # Return the average ECE
    return np.mean(ece_values)

def ece_chart(test_probs, test_labels, n_classes, n_bins=10, fig_name="Calibration Plot.png"):
    """
    Draw Expected Calibration Error (ECE) for each class
    
    Parameters:
    - test_probs (np.ndarray): Array of predicted probabilities for the positive class.
    - test_labels (np.ndarray): Array of true labels (0 or 1).
    - n_classes (int): Number of bins to discretize the predicted probabilities.
    
    Returns:
    - ece (float): The Expected Calibration Error.
    """
    # Create figure
    plt.figure(figsize=(10, 6))

    # Calculate ECE for each class
    ece_values = []

    for class_idx in range(n_classes):
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(
            (test_labels == class_idx).astype(int),
            test_probs[:, class_idx],
            n_bins=n_bins
        )
        plt.plot(prob_pred, prob_true, label=f'Class {class_idx}')
        
        # Calculate ECE for this class
        class_ece = calculate_ece(test_probs, test_labels, class_idx, logits=False, n_bins=n_bins)
        ece_values.append(class_ece)

    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability', fontsize=18)
    plt.ylabel('Fraction of positives', fontsize=18)

    # Add average ECE to title
    avg_ece = np.mean(ece_values)
    plt.title(f'Calibration Plot (Avg. ECE: {avg_ece:.4f})', fontsize=20)
    
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(fig_name, dpi=600, bbox_inches='tight')
    plt.show()


def ece_chart_one_class(test_probs, test_labels, class_idx, n_bins=10, fig_name="Calibration Plot.png"):
    """
    Draw Expected Calibration Error (ECE) for each class
    
    Parameters:
    - test_probs (np.ndarray): Array of predicted probabilities for the positive class.
    - test_labels (np.ndarray): Array of true labels (0 or 1).
    - n_classes (int): Number of bins to discretize the predicted probabilities.
    
    Returns:
    - ece (float): The Expected Calibration Error.
    """
    plt.figure(figsize=(10, 6))
    prob_true, prob_pred = calibration_curve(
        (test_labels == class_idx).astype(int),
        test_probs[:, class_idx],
        n_bins=n_bins
    )
    plt.plot(prob_pred, prob_true, label=f'Class {class_idx}')

    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.savefig(fig_name)
    plt.show()

def draw_multiple_ece_charts(probs_list, labels, n_classes, names, fig_name=None, n_bins=10):
    """
    Draw multiple ECE charts in a single figure for comparison.
    
    Args:
        probs_list (list): List of probability arrays to compare
        labels (numpy.ndarray): Ground truth labels
        n_classes (int): Number of classes
        names (list): List of names for each probability array
        fig_name (str, optional): File name to save the figure
        n_bins (int, optional): Number of bins for calibration
    """
    # Create a figure with subplots in a single row
    fig, axes = plt.subplots(1, len(probs_list), figsize=(30, 6))
    
    # Ensure axes is always an array, even with a single subplot
    if len(probs_list) == 1:
        axes = [axes]
    
    # Define colors for consistent class representation across plots
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    # Plot each method in its own subplot
    for i, (probs, name, ax) in enumerate(zip(probs_list, names, axes)):
        ece_values = []
        
        # Plot calibration curve for each class
        for class_idx in range(n_classes):
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(
                (labels == class_idx).astype(int),
                probs[:, class_idx],
                n_bins=n_bins
            )
            
            # Plot with consistent color for each class
            ax.plot(
                prob_pred, 
                prob_true, 
                marker='o', 
                markersize=4,
                linestyle='-', 
                linewidth=4,
                color=colors[class_idx],
                label=f'Class {class_idx}'
            )
            
            # Calculate ECE for this class
            class_ece = calculate_ece(probs, labels, class_idx, logits=False, n_bins=n_bins)
            ece_values.append(class_ece)
        
        # Add perfect calibration reference line
        ax.plot([0, 1], [0, 1], 'k:', linewidth=4, label='Perfect calibration')
        
        # Set labels and title
        ax.set_xlabel('Mean predicted probability', fontsize=25)
        ax.set_ylabel('Fraction of positives', fontsize=25)
        
        # Add average ECE to title
        avg_ece = np.mean(ece_values)
        ax.set_title(f'{name}\nAvg. ECE: {avg_ece:.4f}', fontsize=30)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Only show legend on the last subplot to save space
        if i == len(probs_list) - 1:
            ax.legend(fontsize=17, loc='best')
        
        ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure if filename is provided
    if fig_name:
        plt.savefig(fig_name, dpi=600, format='pdf', bbox_inches='tight')
    
    plt.show()
    
    # Return average ECE values for each method
    return [np.mean([calculate_ece(probs, labels, c, logits=False, n_bins=n_bins) 
                    for c in range(n_classes)]) 
            for probs in probs_list]