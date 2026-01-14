import numpy as np

def evaluate_accuracy(outputs, labels):
    """
    Evaluate the accuracy of a GNN's predictions.

    Parameters:
    outputs (numpy.ndarray): Outputs returned by the GNN of shape [num_nodes, num_classes].
    labels (numpy.ndarray): Ground truth labels of shape [num_nodes].

    Returns:
    float: Accuracy of the predictions as a value between 0 and 1.
    """
    # Check if outputs and labels have numpy type
    if not isinstance(outputs, np.ndarray) or not isinstance(labels, np.ndarray):
        raise ValueError("Input arrays must be of type numpy.ndarray.")
    
    # Check if outputs and labels have the same number of elements
    if outputs.shape[0] != labels.shape[0]:
        raise ValueError("Input arrays must have the same number of elements.")
    
    # Get the predicted labels by taking the argmax of outputs along the class dimension
    predicted_labels = np.argmax(outputs, axis=1)

    # Compare predicted labels with true labels
    correct_predictions = np.sum(predicted_labels == labels)

    # Calculate accuracy
    accuracy = correct_predictions / labels.shape[0]

    return accuracy