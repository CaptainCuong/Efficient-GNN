"""
Calibrated Random Attack (Calib_Random)

A random-based calibration attack that randomly selects edges to flip for
manipulating model confidence while preserving predicted labels.
This serves as a baseline attack for calibration manipulation.
"""

import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
import random
from .base_attack import BaseAttack
from .calib_attack_loss import (
    kl_divergence_with_uniform,
    underconfidence_objective,
    overconfidence_objective
)


class Calib_Random(BaseAttack):
    """
    Calibrated Random Attack

    This class implements a baseline random attack for calibration manipulation.
    It randomly selects edges (or features) to perturb and applies only those
    perturbations that improve the calibration objective while preserving accuracy.

    Parameters
    ----------
    model : torch.nn.Module
        The surrogate model to attack
    attack_structure : bool, optional
        Whether to attack graph structure (default: True)
    attack_features : bool, optional
        Whether to attack node features (default: False)
    device : str, optional
        Device to run computations on ('cpu' or 'cuda', default: 'cpu')
    """

    def __init__(self, model, attack_structure=True, attack_features=False, device='cpu'):
        super(Calib_Random, self).__init__(model, attack_structure=attack_structure,
                                         attack_features=attack_features, device=device)

    def attack(self, ori_features, ori_adj, target_node, n_perturbations, strategy='under',
               max_trials=100, verbose=False, **kwargs):
        """
        Performs a calibrated random attack on the graph (CUDA-accelerated).

        Randomly selects edges connected to the target node and applies perturbations
        that improve the calibration objective while preserving the predicted label.

        Parameters
        ----------
        ori_features : torch.Tensor or scipy.sparse matrix
            Original node feature matrix of shape (n_nodes, n_features)
        ori_adj : torch.Tensor or scipy.sparse matrix
            Original adjacency matrix of shape (n_nodes, n_nodes)
        target_node : int
            Index of the target node to attack
        n_perturbations : int
            Maximum number of perturbations to apply
        strategy : str, optional
            Attack strategy - 'under': underconfidence, 'over': overconfidence,
            'under_kl': KL-divergence with uniform (default: 'under')
        max_trials : int, optional
            Maximum number of random trials per perturbation (default: 100)
        verbose : bool, optional
            Print detailed attack progress (default: False)
        **kwargs : dict
            Additional arguments including 'n_perturb' and 'best_conf' lists for tracking results

        Returns
        -------
        None
            Modifies self.modified_adj and self.modified_features with adversarial graph
        """

        # Set model to evaluation mode
        self.surrogate.eval()

        # Convert inputs to tensors and keep on GPU
        adj_tensor, features_tensor = self._prepare_inputs(ori_adj, ori_features)

        # Keep everything as tensors on device for speed
        modified_adj = adj_tensor.clone()
        modified_features = features_tensor.clone()
        best_adj = modified_adj.clone()
        best_features = modified_features.clone()

        # Get initial predictions and setup
        with torch.no_grad():
            original_output = self.surrogate(features_tensor, adj_tensor)[[target_node]]
        original_label = original_output.argmax(1)
        best_confidence = F.softmax(original_output, dim=1)[0, original_label].item()

        # Select comparison function based on strategy
        if strategy == 'under' or strategy == 'under_kl':
            is_better = lambda new_conf, best_conf: new_conf < best_conf
            strategy_desc = "underconfidence"
        elif strategy == 'over':
            is_better = lambda new_conf, best_conf: new_conf > best_conf
            strategy_desc = "overconfidence"
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Supported: 'under', 'over', 'under_kl'")

        # Print attack initialization info
        if verbose:
            print("-" * 25, "  CALIB_RANDOM ATTACK BEGINS  ", "-" * 25)
            print(f'Target Node: {target_node}')
            print(f'Number of perturbations: {n_perturbations}')
            print(f'Strategy: {strategy} ({strategy_desc})')
            print(f'Max trials per perturbation: {max_trials}')
            print(f"Before Attack Label: {original_label[0].item()}")
            print(f"Before Attack Confidence: {best_confidence:.4f}")
            print("-" * 50)

        attack_times = 0
        n_nodes = adj_tensor.shape[0]

        # Main attack loop - all operations on GPU
        for perturbation_step in range(n_perturbations):
            found_improvement = False

            # Try multiple random perturbations
            for trial in range(max_trials):
                # Create trial copies (on GPU)
                trial_adj = modified_adj.clone()
                trial_features = modified_features.clone()

                # Randomly choose perturbation type
                if self.attack_structure and self.attack_features:
                    perturb_structure = random.choice([True, False])
                elif self.attack_structure:
                    perturb_structure = True
                else:
                    perturb_structure = False

                if perturb_structure:
                    # Random edge perturbation focused on target node (GPU)
                    success, perturbation_info = self._random_target_edge_perturbation_tensor(
                        trial_adj, target_node, n_nodes
                    )
                else:
                    # Random feature perturbation on target node (GPU)
                    success, perturbation_info = self._random_target_feature_perturbation_tensor(
                        trial_features, target_node
                    )

                if not success:
                    continue

                # Evaluate perturbation (all on GPU)
                with torch.no_grad():
                    new_output = self.surrogate(trial_features, trial_adj)[[target_node]]

                new_label = new_output.argmax(dim=1)

                # Check label preservation
                if new_label != original_label:
                    continue

                # Check calibration improvement
                current_confidence = F.softmax(new_output, dim=1)[0, new_label].item()

                if is_better(current_confidence, best_confidence):
                    # Found improvement
                    found_improvement = True
                    best_confidence = current_confidence
                    best_adj = trial_adj.clone()
                    best_features = trial_features.clone()
                    modified_adj = trial_adj
                    modified_features = trial_features
                    attack_times += 1

                    if verbose:
                        action = perturbation_info['action']
                        if perturb_structure:
                            print(f"Step {perturbation_step + 1}: {action} edge ({target_node}, {perturbation_info['target']})")
                        else:
                            print(f"Step {perturbation_step + 1}: {action} feature ({target_node}, {perturbation_info['target']})")
                        print(f"New Confidence: {current_confidence:.4f}")
                        print(f"Best Confidence: {best_confidence:.4f}")

                    break

            if not found_improvement and verbose:
                print(f"Step {perturbation_step + 1}: No improvement found after {max_trials} trials")

        # Store results
        kwargs.setdefault("n_perturb", []).append(attack_times)
        kwargs.setdefault("best_conf", []).append(best_confidence)

        # Store final results (keep on GPU)
        self.modified_adj = best_adj
        self.modified_features = best_features

        if verbose:
            print("-" * 50)
            print(f"Attack completed with {attack_times} perturbations")
            print(f"Final confidence: {best_confidence:.4f}")
            print("-" * 50)

    def _prepare_inputs(self, adj, features):
        """
        Convert inputs to tensor format for model evaluation.

        Parameters
        ----------
        adj : torch.Tensor or scipy.sparse matrix
            Adjacency matrix
        features : torch.Tensor or scipy.sparse matrix
            Feature matrix

        Returns
        -------
        adj_tensor : torch.Tensor
            Adjacency matrix as tensor
        features_tensor : torch.Tensor
            Feature matrix as tensor
        """
        if sp.issparse(adj):
            adj_tensor = torch.tensor(adj.todense(), dtype=torch.float32, device=self.device)
        else:
            if torch.is_tensor(adj):
                adj_tensor = adj.float().to(self.device)
            else:
                adj_tensor = torch.tensor(adj, dtype=torch.float32, device=self.device)

        if sp.issparse(features):
            features_tensor = torch.tensor(features.todense(), dtype=torch.float32, device=self.device)
        else:
            if torch.is_tensor(features):
                features_tensor = features.float().to(self.device)
            else:
                features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        return adj_tensor, features_tensor

    def _random_target_edge_perturbation(self, adj, target_node):
        """
        Apply a random edge perturbation focused on the target node.

        Parameters
        ----------
        adj : scipy.sparse.lil_matrix
            Adjacency matrix to modify
        target_node : int
            Target node index

        Returns
        -------
        success : bool
            Whether perturbation was successfully applied
        perturbation_info : dict
            Information about the perturbation applied
        """
        n_nodes = adj.shape[0]

        # Get current edges and non-edges for target node
        current_row = adj[target_node].todense().A1
        connected_nodes = np.where(current_row > 0)[0]
        disconnected_nodes = np.where(current_row == 0)[0]

        # Remove self-loops from consideration
        disconnected_nodes = disconnected_nodes[disconnected_nodes != target_node]

        # Randomly choose to add or remove edge
        possible_additions = len(disconnected_nodes) > 0
        possible_removals = len(connected_nodes) > 0

        if not (possible_additions or possible_removals):
            return False, {}

        if possible_additions and possible_removals:
            add_edge = random.choice([True, False])
        elif possible_additions:
            add_edge = True
        else:
            add_edge = False

        if add_edge:
            # Add random edge
            candidate_node = random.choice(disconnected_nodes)
            adj[target_node, candidate_node] = 1
            adj[candidate_node, target_node] = 1
            action = "Add"
        else:
            # Remove random edge
            candidate_node = random.choice(connected_nodes)
            adj[target_node, candidate_node] = 0
            adj[candidate_node, target_node] = 0
            action = "Remove"

        return True, {"action": action, "target": candidate_node}

    def _random_target_feature_perturbation(self, features, target_node):
        """
        Apply a random feature perturbation on the target node.

        Parameters
        ----------
        features : scipy.sparse.lil_matrix
            Feature matrix to modify
        target_node : int
            Target node index

        Returns
        -------
        success : bool
            Whether perturbation was successfully applied
        perturbation_info : dict
            Information about the perturbation applied
        """
        n_features = features.shape[1]

        if n_features == 0:
            return False, {}

        # Randomly select a feature to flip
        feature_idx = random.randint(0, n_features - 1)
        current_value = features[target_node, feature_idx]

        # Flip the feature value (0->1 or 1->0)
        new_value = 1 - current_value
        features[target_node, feature_idx] = new_value

        action = "Add" if new_value > current_value else "Remove"

        return True, {"action": action, "target": feature_idx}

    def _random_target_edge_perturbation_tensor(self, adj, target_node, n_nodes):
        """
        Apply a random edge perturbation focused on the target node (GPU version).

        Parameters
        ----------
        adj : torch.Tensor
            Adjacency matrix to modify (on GPU)
        target_node : int
            Target node index
        n_nodes : int
            Number of nodes in the graph

        Returns
        -------
        success : bool
            Whether perturbation was successfully applied
        perturbation_info : dict
            Information about the perturbation applied
        """
        # Get current edges and non-edges for target node
        current_row = adj[target_node]
        connected_mask = current_row > 0
        disconnected_mask = current_row == 0

        # Remove self-loops from consideration
        disconnected_mask[target_node] = False

        connected_nodes = torch.where(connected_mask)[0]
        disconnected_nodes = torch.where(disconnected_mask)[0]

        # Randomly choose to add or remove edge
        possible_additions = len(disconnected_nodes) > 0
        possible_removals = len(connected_nodes) > 0

        if not (possible_additions or possible_removals):
            return False, {}

        if possible_additions and possible_removals:
            add_edge = random.choice([True, False])
        elif possible_additions:
            add_edge = True
        else:
            add_edge = False

        if add_edge:
            # Add random edge
            idx = random.randint(0, len(disconnected_nodes) - 1)
            candidate_node = disconnected_nodes[idx].item()
            adj[target_node, candidate_node] = 1
            adj[candidate_node, target_node] = 1
            action = "Add"
        else:
            # Remove random edge
            idx = random.randint(0, len(connected_nodes) - 1)
            candidate_node = connected_nodes[idx].item()
            adj[target_node, candidate_node] = 0
            adj[candidate_node, target_node] = 0
            action = "Remove"

        return True, {"action": action, "target": candidate_node}

    def _random_target_feature_perturbation_tensor(self, features, target_node):
        """
        Apply a random feature perturbation on the target node (GPU version).

        Parameters
        ----------
        features : torch.Tensor
            Feature matrix to modify (on GPU)
        target_node : int
            Target node index

        Returns
        -------
        success : bool
            Whether perturbation was successfully applied
        perturbation_info : dict
            Information about the perturbation applied
        """
        n_features = features.shape[1]

        if n_features == 0:
            return False, {}

        # Randomly select a feature to flip
        feature_idx = random.randint(0, n_features - 1)
        current_value = features[target_node, feature_idx].item()

        # Flip the feature value (0->1 or 1->0)
        new_value = 1 - current_value
        features[target_node, feature_idx] = new_value

        action = "Add" if new_value > current_value else "Remove"

        return True, {"action": action, "target": feature_idx}