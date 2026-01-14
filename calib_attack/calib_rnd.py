"""
Calibrated Random Attack (Calib_RND)

Adapted from DeepRobust's RND attack for calibration manipulation.
Instead of just adding random edges to flip predictions, this attack performs
random perturbations while preserving accuracy and optimizing for calibration objectives.

Based on: "Adversarial Attacks on Neural Networks for Graph Data" (KDD'19)
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
import random

from .base_attack import BaseAttack
from .calib_attack_loss import (
    kl_divergence_with_uniform,
    underconfidence_objective,
    overconfidence_objective
)


class Calib_RND(BaseAttack):
    """
    Calibrated Random Attack (Calib_RND)

    This class implements a calibration attack using random perturbations to the graph
    structure and/or features. Unlike the original RND which aims to flip predictions,
    this attack performs random modifications while preserving predicted labels and
    optimizing for calibration objectives.

    The attack randomly samples potential modifications and applies those that improve
    the calibration objective while maintaining prediction accuracy.

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

    Examples
    --------
    >>> from calib_attack.dataset import Dataset
    >>> from calib_attack.gcn import SRG_GCN
    >>> from calib_attack.calib_rnd import Calib_RND
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Surrogate model
    >>> surrogate = SRG_GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                            nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> target_node = 0
    >>> model = Calib_RND(surrogate, attack_structure=True, attack_features=False, device='cpu')
    >>> # Attack
    >>> model.attack(features, adj, target_node, n_perturbations=5, strategy='under_kl')
    >>> modified_adj = model.modified_adj
    """

    def __init__(self, model, attack_structure=True, attack_features=False, device='cpu'):
        super(Calib_RND, self).__init__(model, attack_structure, attack_features, device)

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        self.modified_features = None
        self.target_node = None

    def attack(self, ori_features, ori_adj, target_node, n_perturbations, strategy='under_kl',
               max_trials=100, verbose=False, **kwargs):
        """
        Performs a calibrated random attack on the graph.

        This method randomly samples potential perturbations and applies those that
        improve the calibration objective while preserving the predicted label.

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
            'under_kl': KL-divergence with uniform (default: 'under_kl')
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
        self.target_node = target_node

        # Convert inputs to appropriate format
        if sp.issparse(ori_adj):
            modified_adj = ori_adj.tolil()
        else:
            modified_adj = sp.lil_matrix(ori_adj)

        if sp.issparse(ori_features):
            modified_features = ori_features.tolil()
        else:
            modified_features = sp.lil_matrix(ori_features)

        # Convert to tensors for model evaluation
        def to_tensor_format(adj, features):
            adj_dense = torch.FloatTensor(adj.todense()).to(self.device)
            features_dense = torch.FloatTensor(features.todense()).to(self.device)
            return adj_dense, features_dense

        # Get initial predictions and setup
        adj_tensor, features_tensor = to_tensor_format(modified_adj, modified_features)
        original_output = self.surrogate(features_tensor, adj_tensor, test_idx=target_node).detach()[[target_node]]
        original_label = original_output.argmax(1)
        best_confidence = F.softmax(original_output, dim=1)[0, original_label].item()

        # Select loss function based on strategy
        if strategy == 'under':
            criterion = underconfidence_objective
            is_better = lambda new_conf, best_conf: new_conf < best_conf
        elif strategy == 'over':
            criterion = overconfidence_objective
            is_better = lambda new_conf, best_conf: new_conf > best_conf
        elif strategy == 'under_kl':
            criterion = kl_divergence_with_uniform
            is_better = lambda new_conf, best_conf: new_conf < best_conf
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Print attack initialization info
        if verbose:
            print("-" * 25, "  CALIB_RND ATTACK BEGINS  ", "-" * 25)
            print(f'Target Node: {target_node}')
            print(f'Number of perturbations: {n_perturbations}')
            print(f'Strategy: {strategy}')
            print(f'Max trials per perturbation: {max_trials}')
            print(f"Before Attack Label: {original_label[0].item()}")
            print(f"Before Attack Confidence: {best_confidence:.4f}")
            print("-" * 50)

        # Track best configuration
        best_adj = modified_adj.copy()
        best_features = modified_features.copy()
        attack_times = 0

        # Apply perturbations iteratively
        for perturbation_step in range(n_perturbations):
            found_improvement = False
            current_adj = modified_adj.copy()
            current_features = modified_features.copy()

            # Try multiple random perturbations
            for trial in range(max_trials):
                # Reset to current state
                trial_adj = current_adj.copy()
                trial_features = current_features.copy()

                # Randomly choose perturbation type
                if self.attack_structure and self.attack_features:
                    perturb_structure = random.choice([True, False])
                elif self.attack_structure:
                    perturb_structure = True
                else:
                    perturb_structure = False

                if perturb_structure:
                    # Random edge perturbation
                    success, perturbation_info = self._random_edge_perturbation(
                        trial_adj, target_node
                    )
                else:
                    # Random feature perturbation
                    success, perturbation_info = self._random_feature_perturbation(
                        trial_features, target_node
                    )

                if not success:
                    continue

                # Evaluate perturbation
                trial_adj_tensor, trial_features_tensor = to_tensor_format(trial_adj, trial_features)

                with torch.no_grad():
                    new_output = self.surrogate(trial_features_tensor, trial_adj_tensor, test_idx=target_node)[[target_node]]

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
                    best_adj = trial_adj.copy()
                    best_features = trial_features.copy()
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

        # Convert back to appropriate format and store
        self.modified_adj = sp.csr_matrix(best_adj)
        self.modified_features = sp.csr_matrix(best_features)
        self.check_adj(best_adj.todense())

        if verbose:
            print("-" * 50)
            print(f"Attack completed with {attack_times} perturbations")
            print(f"Final confidence: {best_confidence:.4f}")
            print("-" * 50)

    def _random_edge_perturbation(self, adj, target_node):
        """
        Apply a random edge perturbation (add or remove edge).

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

    def _random_feature_perturbation(self, features, target_node):
        """
        Apply a random feature perturbation (flip feature value).

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

    def random_node_injection(self, ori_features, ori_adj, target_node, n_injected=1,
                            n_perturbations=5, strategy='under_kl', max_trials=50, verbose=False, **kwargs):
        """
        Perform calibration attack by injecting random nodes.

        This method adds fake nodes to the graph and connects them strategically
        to manipulate the target node's calibration while preserving its label.

        Parameters
        ----------
        ori_features : torch.Tensor or scipy.sparse matrix
            Original node feature matrix
        ori_adj : torch.Tensor or scipy.sparse matrix
            Original adjacency matrix
        target_node : int
            Index of the target node to attack
        n_injected : int, optional
            Number of nodes to inject (default: 1)
        n_perturbations : int, optional
            Number of edges to add for each injected node (default: 5)
        strategy : str, optional
            Attack strategy (default: 'under_kl')
        max_trials : int, optional
            Maximum trials for finding good connections (default: 50)
        verbose : bool, optional
            Print detailed progress (default: False)
        **kwargs : dict
            Additional arguments for tracking results

        Returns
        -------
        None
            Modifies self.modified_adj and self.modified_features with injected nodes
        """

        self.surrogate.eval()
        self.target_node = target_node

        # Convert to appropriate format
        if sp.issparse(ori_adj):
            N = ori_adj.shape[0]
        else:
            N = ori_adj.shape[0]

        if sp.issparse(ori_features):
            D = ori_features.shape[1]
        else:
            D = ori_features.shape[1]

        # Expand adjacency matrix and features
        modified_adj = self._reshape_mx(ori_adj, shape=(N + n_injected, N + n_injected))
        modified_features = self._reshape_mx(ori_features, shape=(N + n_injected, D))

        # Get initial state
        def to_tensor_format(adj, features):
            adj_dense = torch.FloatTensor(adj.todense()).to(self.device)
            features_dense = torch.FloatTensor(features.todense()).to(self.device)
            return adj_dense, features_dense

        adj_tensor, features_tensor = to_tensor_format(modified_adj, modified_features)
        original_output = self.surrogate(features_tensor, adj_tensor, test_idx=target_node).detach()[[target_node]]
        original_label = original_output.argmax(1)
        best_confidence = F.softmax(original_output, dim=1)[0, original_label].item()

        # Select comparison function
        if strategy in ['under', 'under_kl']:
            is_better = lambda new_conf, best_conf: new_conf < best_conf
        else:
            is_better = lambda new_conf, best_conf: new_conf > best_conf

        if verbose:
            print(f"Injecting {n_injected} nodes with {n_perturbations} connections each")
            print(f"Initial confidence: {best_confidence:.4f}")

        best_adj = modified_adj.copy()
        best_features = modified_features.copy()
        attack_times = 0

        # Inject nodes one by one
        for fake_node_idx in range(N, N + n_injected):
            # Connect fake node to target node
            modified_adj[fake_node_idx, target_node] = 1
            modified_adj[target_node, fake_node_idx] = 1

            # Copy features from a random existing node
            source_node = random.randint(0, N - 1)
            modified_features[fake_node_idx] = modified_features[source_node]

            # Try to find good connections for the fake node
            for _ in range(max_trials):
                trial_adj = modified_adj.copy()

                # Randomly connect to other nodes
                potential_connections = list(range(N))
                potential_connections.remove(target_node)  # Already connected
                random.shuffle(potential_connections)

                connected_nodes = []
                for node in potential_connections[:n_perturbations]:
                    trial_adj[fake_node_idx, node] = 1
                    trial_adj[node, fake_node_idx] = 1
                    connected_nodes.append(node)

                # Evaluate configuration
                trial_adj_tensor, trial_features_tensor = to_tensor_format(trial_adj, modified_features)

                with torch.no_grad():
                    new_output = self.surrogate(trial_features_tensor, trial_adj_tensor, test_idx=target_node)[[target_node]]

                new_label = new_output.argmax(dim=1)

                if new_label != original_label:
                    continue

                current_confidence = F.softmax(new_output, dim=1)[0, new_label].item()

                if is_better(current_confidence, best_confidence):
                    best_confidence = current_confidence
                    best_adj = trial_adj.copy()
                    best_features = modified_features.copy()
                    modified_adj = trial_adj
                    attack_times += 1

                    if verbose:
                        print(f"Injected node {fake_node_idx} connected to {connected_nodes}")
                        print(f"New confidence: {current_confidence:.4f}")
                    break

        # Store results
        kwargs.setdefault("n_perturb", []).append(attack_times)
        kwargs.setdefault("best_conf", []).append(best_confidence)

        self.modified_adj = sp.csr_matrix(best_adj)
        self.modified_features = sp.csr_matrix(best_features)
        self.check_adj(best_adj.todense())

        if verbose:
            print(f"Node injection completed. Final confidence: {best_confidence:.4f}")

    def _reshape_mx(self, mx, shape):
        """Reshape sparse matrix to new dimensions."""
        if sp.issparse(mx):
            indices = mx.nonzero()
            return sp.csr_matrix((mx.data, (indices[0], indices[1])), shape=shape).tolil()
        else:
            # Handle dense matrices/arrays
            old_shape = mx.shape
            new_mx = np.zeros(shape)
            new_mx[:old_shape[0], :old_shape[1]] = mx
            return sp.lil_matrix(new_mx)