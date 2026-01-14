import torch
import torch.nn as nn
import torch.nn.functional as F

# Assume `model` is your pre-trained GNN model and `data` is a batch of graph data
# `logits` are the outputs of the GNN model before applying softmax
# `labels` are the true labels of the nodes

def distance_from_uniform(logits, labels):
    # Calculate softmax probabilities
    softmax_probs = F.softmax(logits, dim=1)
    
    # Get the index of the highest probability for each sample in the batch
    max_prob_indices = torch.argmax(softmax_probs, dim=1)
    
    # Create a mask to exclude the highest probability element for each sample
    mask = torch.ones_like(softmax_probs, dtype=bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    
    # Extract the remaining probabilities for each sample
    remaining_probs = softmax_probs[mask].reshape(logits.shape[0], -1)
    
    # Create the uniform distribution tensor for the remaining classes
    num_remaining_classes = remaining_probs.shape[1]
    uniform_probs = torch.full((logits.shape[0], num_remaining_classes), 1.0 / (num_remaining_classes + 1))
    
    # Calculate the L2 distance (Euclidean norm) for each sample
    distances = torch.norm(remaining_probs - uniform_probs, p=2, dim=1)
    
    return - distances

def maximize_minimum_softmax(logits, labels=None):
    # Calculate softmax probabilities
    softmax_probs = F.softmax(logits, dim=1)
    
    # Find the minimum probability for each sample in the batch
    min_probs, _ = torch.min(softmax_probs, dim=1)
    
    # Objective function: maximize the sum of the minimum probabilities
    # Negate it to use with gradient-based optimizers (as they minimize the loss)
    objective = torch.sum(min_probs)
    
    return objective

def kl_divergence_remaining_with_uniform(logits, labels):
    # Calculate softmax probabilities
    softmax_probs = F.softmax(logits, dim=1)
    
    # Get the index of the highest probability for each sample in the batch
    max_prob_indices = torch.argmax(softmax_probs, dim=1)

    # Create a mask to exclude the highest probability element for each sample
    mask = torch.ones_like(softmax_probs, dtype=bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
  
    # Extract the remaining probabilities for each sample
    remaining_probs = softmax_probs[mask].reshape(logits.shape[0], -1)
    
    # Create the uniform distribution tensor for the remaining classes
    num_remaining_classes = remaining_probs.shape[1]
    uniform_probs = torch.full((logits.shape[0], num_remaining_classes), 1.0 / num_remaining_classes)
    
    # Calculate KL Divergence for each sample in the batch
    kl_div = - F.kl_div(remaining_probs.log(), uniform_probs, reduction='batchmean')
    
    return kl_div

def kl_divergence_with_uniform(logits, labels):
    # Calculate softmax probabilities
    softmax_probs = F.softmax(logits, dim=1)
    
    # Create the uniform distribution tensor
    num_classes = logits.shape[1]
    uniform_probs = torch.full_like(softmax_probs, 1.0 / num_classes)
    
    # Calculate KL Divergence
    kl_div = - F.kl_div(softmax_probs.log(), uniform_probs, reduction='batchmean')
    
    return kl_div  # Negate for maximization in the context of an attack

# def adaptive_kl_loss(logits, labels):
#     # Calculate softmax probabilities
#     softmax_probs = F.softmax(logits, dim=1)
    
#     # Create the uniform distribution tensor
#     num_classes = logits.shape[1]
#     uniform_probs = torch.full_like(softmax_probs, 1.0 / num_classes)
    
#     # Calculate KL Divergence
#     kl_div = - F.kl_div(softmax_probs.log(), uniform_probs, reduction='batchmean')

#     # Adaptive scaling factor (based on the highest confidence in the predictions)
#     max_confidence, _ = logits.max(dim=1)  # Shape: (batch_size,)
#     scaling_factor = torch.exp(max_confidence)  # Shape: (batch_size,)

#     # Apply scaling factor to the KL loss
#     adaptive_kl_loss = (scaling_factor * kl_div).mean()
    
#     return adaptive_kl_loss

def kl_divergence_target(logits, target_label, res_gt):
    """
    Calculates the KL divergence between the softmax probabilities of the logits and a target distribution.

    This function computes a target distribution based on the predicted labels, target labels, and ground truth,
    then calculates the KL divergence between the softmax probabilities of the input logits and this target distribution.

    Args:
        logits (torch.Tensor): The raw output scores from the model, typically of shape [batch_size, num_classes].
        target_label (torch.Tensor): The target labels for each sample in the batch, typically of shape [batch_size].
        res_gt (torch.Tensor): The ground truth labels for each sample in the batch, typically of shape [batch_size].

    Returns:
        torch.Tensor: A scalar tensor representing the KL divergence between the softmax probabilities
                      of the logits and the computed target distribution.
    """
    # Calculate softmax probabilities
    softmax_probs = F.softmax(logits, dim=1)

    # Get the predicted labels
    predicted_labels = torch.argmax(softmax_probs, dim=1)

    # Create a tensor to store the target distribution
    target_dist = torch.zeros_like(softmax_probs)

    # Iterate through each sample in the batch
    for i in range(logits.shape[0]):
        if predicted_labels[i] == target_label[i]: # Predicted Positive at the target label
            if res_gt[i] == target_label[i]:
                # Prediction is positive and groundtruth is target label
                # Use uniform distribution
                target_dist[i] = torch.full((logits.shape[1],), 1.0 / logits.shape[1])
            else:
                # Prediction is positive but groundtruth is not target label
                # Set target label to 1, others to 0
                target_dist[i, target_label[i]] = 1.0
        else: # Predicted Negative at the target label
            if res_gt[i] != target_label[i]:
                # Prediction is negative and groundtruth is not target label
                # Set target label to 0, distribute equally among others
                num_other_classes = logits.shape[1] - 1
                target_dist[i] = torch.full((logits.shape[1],), 1.0 / num_other_classes)
                target_dist[i, target_label[i]] = 0.0
            else:
                # Prediction is negative and groundtruth is target label
                # Set target label and highest probability to 0.5, others to 0
                highest_prob_index = predicted_labels[i]
                target_dist[i, target_label[i]] = 0.5
                target_dist[i, highest_prob_index] = 0.5

    # Calculate KL Divergence
    kl_div = F.kl_div(softmax_probs.log(), target_dist, reduction='batchmean')

    return - kl_div
    

# Underconfidence objective
def underconfidence_objective(logits, labels):
    # Apply softmax to obtain probabilities
    probs = F.softmax(logits, dim=1)

    # Get the probabilities of the predicted classes
    predicted_probs = probs[torch.arange(len(labels)), labels]

    # Create a mask that sets the true class probabilities to a very low value
    mask = torch.ones_like(probs)
    mask[torch.arange(len(labels)), labels] = 0

    # Set the true class probabilities to a very low value
    masked_probs = probs * mask

    # Find the maximum probability among the other classes
    max_other_probs, _ = torch.max(masked_probs, dim=1)

    # Underconfidence objective: maximize the gap between the true class and the most likely other class
    underconfidence_loss = -torch.mean(predicted_probs - max_other_probs)

    return underconfidence_loss

# Overconfidence objective
def overconfidence_objective(logits, labels):
    """
    Calculates the overconfidence objective loss.

    This function computes a loss that encourages the model to be less confident
    in its predictions by maximizing the difference between 1 and the predicted
    probabilities for the true labels.

    Args:
        logits (torch.Tensor): The raw output scores from the model, typically of
                               shape [batch_size, num_classes].
        labels (torch.Tensor): The true labels for each sample in the batch,
                               typically of shape [batch_size].

    Returns:
        torch.Tensor: A scalar tensor representing the overconfidence loss.
                      Higher values indicate more overconfidence.
    """
    # Apply softmax to obtain probabilities
    probs = F.softmax(logits, dim=1)

    # Get the probabilities of the predicted classes
    predicted_probs = probs[torch.arange(len(labels)), labels]

    # Overconfidence objective: maximize the probability of the true class (minimize 1 - predicted_probs)
    overconfidence_loss = - torch.mean(1 - predicted_probs)

    return overconfidence_loss

# def combined_underconfidence_objective(logits, labels, alpha=0.5):
    
#     classification_loss = F.nll_loss(logits, labels)

#     # Apply softmax to obtain probabilities
#     probs = F.softmax(logits, dim=1)

#     # Get the probabilities of the predicted classes
#     predicted_probs = probs[torch.arange(len(labels)), labels]
    
#     # Overconfidence objective: minimize the probability of the true class (maximize 1 - predicted_probs)
#     overconfidence_loss = torch.mean(1 - predicted_probs)
    
#     attack_loss = overconfidence_loss - alpha * classification_loss

#     return attack_loss

# Maximum Miscalibration objective
def maximum_miscalibration_objective(logits, labels):
    # Apply softmax to obtain probabilities
    probs = F.softmax(logits, dim=1)

    # Get the probabilities of the predicted classes
    predicted_probs = probs[torch.arange(len(labels)), labels]

    # Correctly classified nodes should have low confidence
    miscalibration_loss_correct = -predicted_probs[labels == logits.argmax(dim=1)]

    # Incorrectly classified nodes should have high confidence
    miscalibration_loss_incorrect = predicted_probs[labels != logits.argmax(dim=1)]

    # Combine both losses for maximum miscalibration
    miscalibration_loss = torch.mean(miscalibration_loss_correct) + torch.mean(miscalibration_loss_incorrect)

    return miscalibration_loss

# Random Confidence Attack objective
def random_confidence_objective(logits, labels):
    num_classes = logits.shape[1]

    # Apply softmax to obtain probabilities
    probs = F.softmax(logits, dim=1)

    # Get the probabilities of the predicted classes
    predicted_probs = probs[torch.arange(len(labels)), labels]

    # Generate random target confidence scores between 1/K and 1.0
    random_confidences = torch.rand(len(labels)) * (1.0 - 1.0/num_classes) + 1.0/num_classes

    # Initialize loss
    rca_loss = 0.0

    # Determine if the current confidence should be increased or decreased
    for i in range(len(labels)):
        if predicted_probs[i] < random_confidences[i]:
            # Overconfidence attack: move predicted_probs[i] towards random_confidences[i]
            rca_loss += random_confidences[i] - predicted_probs[i]
        else:
            # Underconfidence attack: move predicted_probs[i] away from random_confidences[i]
            rca_loss += predicted_probs[i] - random_confidences[i]

    # Average the loss over all nodes
    rca_loss = rca_loss / len(labels)

    return rca_loss

# # Example usage
# # logits: Tensor of shape [num_nodes, num_classes]
# # labels: Tensor of shape [num_nodes]
# logits = torch.randn((10, 5), requires_grad=True)  # Example logits
# labels = torch.randint(0, 5, (10,))  # Example labels
# print(logits)
# print(labels)
# # Compute the objectives
# uc_loss = underconfidence_objective(logits, labels)
# oc_loss = overconfidence_objective(logits, labels)
# mma_loss = maximum_miscalibration_objective(logits, labels)
# rca_loss = random_confidence_objective(logits, labels)

# print("Underconfidence Loss:", uc_loss.item())
# print("Overconfidence Loss:", oc_loss.item())
# print("Maximum Miscalibration Loss:", mma_loss.item())
# print("Random Confidence Loss:", rca_loss.item())