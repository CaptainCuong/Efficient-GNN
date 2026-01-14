"""
    Adversarial Examples on Graph Data: Deep Insights into Attack and Defense
        https://arxiv.org/pdf/1903.01610.pdf
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from .base_attack import BaseAttack
from . import utils
from .calib_attack_loss import underconfidence_objective, overconfidence_objective

class Calib_IGA(BaseAttack):
    """Calib_IGA: Calibration attack using Integrated Gradients approach.

    Adapts the IGAttack method to target calibration instead of accuracy.
    Implements both overconfidence and underconfidence calibration attacks.

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph (optional, for compatibility)
    device: str
        'cpu' or 'cuda'
    """

    def __init__(self, model, nnodes=None, device='cpu'):
        super(Calib_IGA, self).__init__(model, attack_structure=True, attack_features=False, device=device)

        self.modified_adj = None
        self.target_node = None

    def attack(self, ori_features, ori_adj, target_node, n_perturbations, strategy, res_gt=None, steps=10, verbose=False, **kwargs):
        """Generate calibration perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph
        strategy : str
            Attack strategy: 'over' for overconfidence, 'under' for underconfidence
        res_gt : torch.Tensor
            Ground truth labels for calibration evaluation
        steps : int
            steps for computing integrated gradients
        verbose : bool
            Whether to print detailed progress
        """

        if res_gt is None:
            raise ValueError("res_gt must be provided for calibration attacks")

        if strategy not in ['over', 'under']:
            raise ValueError("strategy must be 'over' or 'under'")

        self.surrogate.eval()
        self.target_node = target_node

        # Handle input tensors - assume inputs are already torch tensors like in Calib_FGA
        adj = ori_adj.detach().clone().to(self.device)
        features = ori_features.detach().clone().to(self.device)
        res_gt = torch.tensor(res_gt, device=self.device) if not isinstance(res_gt, torch.Tensor) else res_gt.to(self.device)
        
        # Get initial predictions
        original_output = self.surrogate(features, adj).detach()[[target_node]]
        original_label = original_output.argmax(1)
        original_confidence = F.softmax(original_output, dim=1)[0, original_label].item()

        if verbose:
            print(f"[{strategy.upper()}][Node {target_node}] Starting calibration attack")
            print(f"Original label: {original_label.item()}, confidence: {original_confidence:.4f}")
        
        # Calculate importance scores for edges using calibration loss
        importance_scores = self.calc_calibration_importance_edge(
            features, adj, target_node, strategy, steps, verbose
        )
        
        best_adj = adj.clone()
        best_confidence = original_confidence
        attack_times = 0
        
        # Apply perturbations based on importance scores
        for i in range(n_perturbations):
            print(i)
            if len(importance_scores) == 0:
                break

            # Find the edge with highest importance score
            max_idx = np.argmax(importance_scores)

            # Apply edge perturbation
            current_value = adj[target_node, max_idx].item()
            new_value = 1 - current_value  # Flip edge

            # Create a new adjacency matrix to avoid in-place operations
            adj = adj.detach().clone()
            adj[target_node, max_idx] = new_value
            adj[max_idx, target_node] = new_value

            # Check if label is preserved
            with torch.no_grad():
                new_output = self.surrogate(features, adj)[[target_node]]
                new_label = new_output.argmax(dim=1)
                new_confidence = F.softmax(new_output, dim=1)[0, new_label].item()

            if new_label != original_label:
                if verbose:
                    print(f"[{strategy.upper()}][Node {target_node}] Early stop at step {i+1}: label flipped")
                break

            # Update best configuration based on strategy
            attack_times += 1
            if strategy == 'over' and new_confidence >= best_confidence:
                best_confidence = new_confidence
                best_adj = adj.clone()
            elif strategy == 'under' and new_confidence <= best_confidence:
                best_confidence = new_confidence
                best_adj = adj.clone()

            if verbose:
                action = "Added" if new_value > current_value else "Removed"
                print(f"[{strategy.upper()}][Node {target_node}][Step {i+1}] {action} edge to node {max_idx}")
                print(f"Confidence: {original_confidence:.4f} -> {new_confidence:.4f} (Δ {new_confidence - original_confidence:+.4f})")

            # Remove this edge from consideration
            importance_scores[max_idx] = -np.inf
            
        # Store results
        kwargs.get("n_perturb", []).append(attack_times)
        kwargs.get("best_conf", []).append(best_confidence)

        self.modified_adj = best_adj.detach().clone()
        self.check_adj(best_adj.detach().cpu().numpy())

        if verbose:
            final_delta = best_confidence - original_confidence
            print(f"[{strategy.upper()}][Node {target_node}] Attack completed")
            print(f"Perturbations used: {attack_times}/{n_perturbations}")
            print(f"Final confidence: {original_confidence:.4f} -> {best_confidence:.4f} (Δ {final_delta:+.4f})")

    def calc_calibration_importance_edge(self, features, adj, target_node, strategy, steps=10, verbose=False):
        """Calculate integrated gradients for edges using calibration loss.

        Parameters
        ----------
        features : torch.Tensor
            Node features
        adj : torch.Tensor
            Adjacency matrix
        target_node : int
            Target node index
        strategy : str
            'over' for overconfidence, 'under' for underconfidence
        steps : int
            Number of integration steps
        verbose : bool
            Whether to show progress

        Returns
        -------
        numpy.ndarray
            Importance scores for each potential edge
        """

        # Select calibration loss function
        if strategy == 'over':
            criterion = overconfidence_objective
        else:  # strategy == 'under'
            criterion = underconfidence_objective

        # Create a copy for gradient computation
        adj_grad = adj.detach().clone().requires_grad_(True)

        baseline_add = adj_grad.clone()
        baseline_remove = adj_grad.clone()
        baseline_add.data[target_node] = 1
        baseline_remove.data[target_node] = 0
        integrated_grad_list = []
        
        iterator = tqdm(range(adj.shape[1])) if verbose else range(adj.shape[1])
        
        for j in iterator:
            if j == target_node:
                integrated_grad_list.append(0)
                continue

            # Create scaled inputs for integration
            if adj_grad[target_node][j]:
                # Edge exists, integrate from current to baseline_remove
                scaled_inputs = [baseline_remove + (float(k)/steps) * (adj_grad - baseline_remove)
                               for k in range(0, steps + 1)]
            else:
                # Edge doesn't exist, integrate from baseline_add to current
                scaled_inputs = [baseline_add - (float(k)/steps) * (baseline_add - adj_grad)
                               for k in range(0, steps + 1)]

            gradient_sum = 0
            
            for new_adj in scaled_inputs:
                output = self.surrogate(features, new_adj)[[target_node]]
                current_label = output.argmax(dim=1)

                # Use calibration loss instead of classification loss
                loss = criterion(output, current_label)

                grad_result = torch.autograd.grad(loss, adj_grad, retain_graph=True)[0]
                grad_val = grad_result[target_node][j]
                gradient_sum += grad_val

            # Calculate integrated gradient
            if adj_grad[target_node][j]:
                avg_grad = (adj_grad[target_node][j] - 0) * gradient_sum.mean()
            else:
                avg_grad = (1 - adj_grad[target_node][j]) * gradient_sum.mean()

            integrated_grad_list.append(avg_grad.detach().item())
        
        # Convert to numpy and apply perturbation direction
        integrated_grad_list = np.array(integrated_grad_list)
        adj_binary = (adj_grad > 0).detach().cpu().numpy()
        integrated_grad_list = (-2 * adj_binary[target_node] + 1) * integrated_grad_list
        integrated_grad_list[target_node] = -10  # Prevent self-loops

        return integrated_grad_list

