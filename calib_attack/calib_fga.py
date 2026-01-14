import torch
from .base_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
from . import utils
import time
import torch.nn.functional as F
import scipy.sparse as sp
from .calib_attack_loss import underconfidence_objective, overconfidence_objective, \
                                kl_divergence_with_uniform, \
                                maximize_minimum_softmax, distance_from_uniform, \
                                kl_divergence_target, kl_divergence_remaining_with_uniform
import numpy as np
from queue import PriorityQueue
from tabulate import tabulate

class Calib_FGA(BaseAttack):
    """Calib_FGA/FGSM.

    Parameters
    ----------
    model :
        model to attack
    feature_shape : tuple
        shape of the input node features
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'

    """
    
    def __init__(self, model, feature_shape=None, attack_features=False, device='cuda', logits=True):
        """Initialize the calibrated Fast Gradient Attack.

        Parameters
        ----------
        model : torch.nn.Module
            Surrogate network used for gradient computation.
        feature_shape : tuple, optional
            Feature tensor shape if feature perturbations are enabled.
        attack_features : bool, optional
            Whether feature perturbations are allowed (default: False).
        device : str, optional
            Device identifier for intermediate tensors (default: 'cuda').
        logits : bool, optional
            Legacy flag kept for compatibility with original API.
        """

        super(Calib_FGA, self).__init__(model, attack_structure=True, attack_features=attack_features, device=device)

        assert not self.attack_features, "not support attacking features"

        if self.attack_features:
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

    @staticmethod
    def _log_attack_header(strategy, target_node, budget, original_label, original_confidence):
        banner = f"[{strategy.upper()}][Node {target_node}]"
        print("\n" + "=" * 80)
        print(f"{banner} Starting attack | budget={budget}")
        print(
            f"Baseline label={original_label} | confidence={original_confidence:.4f}"
        )
        print("-" * 80)

    @staticmethod
    def _log_iteration(
        strategy,
        target_node,
        step,
        budget,
        action,
        edge_idx,
        prev_confidence,
        new_confidence,
        baseline_confidence,
        predicted_label,
        loss=None,
        log_rows=None,
    ):
        step_delta = new_confidence - prev_confidence
        total_delta = new_confidence - baseline_confidence
        message = (
            f"[{strategy.upper()}][Node {target_node}][Step {step}/{budget}] "
            f"{action} edge ({target_node}, {edge_idx}) | pred {predicted_label} | "
            f"conf {new_confidence:.4f} (Δ step {step_delta:+.4f}, Δ total {total_delta:+.4f})"
        )
        if loss is not None:
            message += f" | loss {loss:.4f}"
        print(message)

        if log_rows is not None:
            row = [
                step,
                action,
                predicted_label,
                f"{new_confidence:.4f}",
                f"{step_delta:+.4f}",
                f"{total_delta:+.4f}",
                f"{loss:.4f}" if loss is not None else "-",
            ]
            log_rows.append(row)

    @staticmethod
    def _log_attack_summary(
        strategy,
        target_node,
        used_perturbations,
        budget,
        original_label,
        final_label,
        baseline_confidence,
        final_confidence,
    ):
        label_status = "preserved" if final_label == original_label else "CHANGED"
        print(f"[{strategy.upper()}][Node {target_node}] Perturbations used: {used_perturbations}/{budget}")
        print(
            f"Label: {original_label} -> {final_label} ({label_status})"
        )
        print(
            f"Confidence: {baseline_confidence:.4f} -> {final_confidence:.4f} "
            f"(Δ {final_confidence - baseline_confidence:+.4f})"
        )
        print("=" * 80 + "\n")

    def attack(
        self,
        ori_features,
        ori_adj,
        target_node,
        n_perturbations,
        strategy,
        *,
        target_label=0,
        res_gt=None,
        verbose=False,
        **kwargs,
    ):
        """Run the baseline calibrated Fast Gradient Attack on a single node.

        Iteratively flips the highest-impact edge while the predicted label remains unchanged,
        recording the configuration that best satisfies the requested calibration objective.

        Parameters
        ----------
        ori_features : torch.Tensor
            Node feature matrix of shape ``(n_nodes, n_features)``.
        ori_adj : torch.Tensor
            Dense adjacency matrix of shape ``(n_nodes, n_nodes)``.
        target_node : int
            Index of the node to attack.
        n_perturbations : int
            Maximum number of edge flips to apply.
        strategy : str
            Strategy keyword: ``'over'``, ``'under'``, ``'under_kl'``, ``'target'``, or ``'max'``.
        res_gt : torch.Tensor, optional
            Ground-truth labels used when evaluating calibration objectives (required).
        target_label : int, optional
            Target class for the ``'target'`` strategy (default: ``0``).
        verbose : bool, optional
            If ``True``, emit per-iteration diagnostics (default: ``False``).
        **kwargs : dict, optional
            Optional trackers such as ``n_perturb`` and ``best_conf`` lists for logging.

        Raises
        ------
        ValueError
            If ``res_gt`` is not provided.

        Returns
        -------
        None
            The best perturbed adjacency is stored on ``self.modified_adj``.
        """
        display_strategy = strategy

        # Get initial predictions and setup
        original_output = self.surrogate(ori_features, ori_adj).detach()[[target_node]]

        if res_gt is None:
            raise ValueError("res_gt must be provided for Calib_FGA attacks")

        res_gt = res_gt[[target_node]]
        original_label = original_output.argmax(1)
        original_label_item = int(original_label.item())
        baseline_probs = F.softmax(original_output, dim=1)
        best_confidence = baseline_probs[0, original_label_item].item()
        initial_confidence = best_confidence

        self._log_attack_header(
            display_strategy,
            target_node,
            n_perturbations,
            original_label_item,
            initial_confidence,
        )

        # Select loss function based on attack strategy
        if strategy == 'over':
            criterion = overconfidence_objective
        elif strategy == 'under':
            criterion = underconfidence_objective
        elif strategy == 'under_kl':
            strategy = "under"
            criterion = kl_divergence_with_uniform
        elif strategy == 'test':
            strategy = "under"
            if "criterion" in kwargs:
                if kwargs["criterion"] == "normal":
                    criterion = underconfidence_objective
                elif kwargs["criterion"] == "kl":
                    criterion = kl_divergence_with_uniform
                else:
                    raise ValueError("criterion must be normal or kl")
            else:
                raise ValueError("criterion must be provided for test strategy")
        elif strategy == 'target':
            criterion = kl_divergence_target
        elif strategy == 'max':
            criterion = kl_divergence_target
            target_label = original_label
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Initialize adjacency matrices
        modified_adj = ori_adj.detach().clone()
        modified_features = ori_features.detach().clone()
        best_adj = modified_adj.detach().clone()

        self.surrogate.eval()

        # Main attack loop
        modified_adj.requires_grad = True
        attack_times = 0

        table_rows = [] if verbose else None

        for i in range(n_perturbations):
            # Forward pass and loss computation
            output = self.surrogate(modified_features, modified_adj)[[target_node]]
            current_label = output.argmax(dim=1)
            prev_confidence = F.softmax(output.detach(), dim=1)[0, current_label].item()

            if self.attack_structure:
                # Compute loss based on strategy
                if strategy != 'target':
                    loss = criterion(output, current_label)
                else:
                    loss = criterion(
                        output,
                        torch.tensor([target_label], device=output.device),
                        res_gt,
                    )

                grad = torch.autograd.grad(loss, modified_adj, allow_unused=True)[0]

                # Apply symmetry balancing and edge flip logic
                grad = (grad[target_node] + grad[:, target_node]) * (
                    -2 * modified_adj[target_node] + 1
                )
                grad[target_node] = -10  # Prevent self-loops
                grad_argmax = torch.argmax(grad)

            # Apply edge perturbation (flip edge: 0→1 or 1→0)
            value = -2*modified_adj[target_node][grad_argmax] + 1
            modified_adj.data[target_node][grad_argmax] += value
            modified_adj.data[grad_argmax][target_node] += value

            # Check for label preservation
            with torch.no_grad():
                new_output = self.surrogate(modified_features, modified_adj)[[target_node]]
            new_label = new_output.argmax(dim=1)

            if new_label != original_label:
                print(
                    f"[{display_strategy.upper()}][Node {target_node}] "
                    f"Early stop at step {i + 1}: label flipped to {int(new_label.item())}"
                )
                break

            # Update best configuration based on strategy
            attack_times += 1
            current_label = new_label
            predicted_label = int(current_label.item())
            current_confidence = F.softmax(new_output, dim=1)[0, predicted_label].item()

            if strategy == 'over' and current_confidence >= best_confidence:
                best_confidence = current_confidence
                best_adj = modified_adj.detach().clone()
            elif strategy == 'under' and current_confidence <= best_confidence:
                best_confidence = current_confidence
                best_adj = modified_adj.detach().clone()
            elif strategy == 'target':
                # Adaptive strategy based on ground truth and prediction alignment
                if (target_label == original_label and target_label == res_gt.item()) or \
                   (target_label != original_label and target_label == res_gt.item()):
                    # Underconfidence attack for correct predictions
                    if current_confidence <= best_confidence:
                        best_confidence = current_confidence
                        best_adj = modified_adj.detach().clone()
                else:
                    # Overconfidence attack for incorrect predictions
                    if current_confidence >= best_confidence:
                        best_confidence = current_confidence
                        best_adj = modified_adj.detach().clone()

            # Print progress information
            if verbose:
                action = "Added" if value > 0 else "Removed"
                self._log_iteration(
                    display_strategy,
                    target_node,
                    i + 1,
                    n_perturbations,
                    action,
                    int(grad_argmax.item()),
                    prev_confidence,
                    current_confidence,
                    initial_confidence,
                    predicted_label,
                    loss.item() if 'loss' in locals() else None,
                    log_rows=table_rows,
                )

        # Store results and finalize
        kwargs["n_perturb"].append(attack_times)
        kwargs["best_conf"].append(best_confidence)

        modified_adj = modified_adj.detach()
        self.check_adj(modified_adj)
        self.modified_adj = best_adj

        if verbose and table_rows:
            headers = ["Step", "Action", "Pred", "Conf", "Δ step", "Δ total", "Loss"]
            print(tabulate(table_rows, headers=headers, tablefmt="grid"))

        with torch.no_grad():
            final_output = self.surrogate(ori_features, self.modified_adj).detach()[[target_node]]
            final_label = int(final_output.argmax(dim=1).item())
            final_confidence = float(
                F.softmax(final_output, dim=1)[0, final_label].item()
            )

        self._log_attack_summary(
            display_strategy,
            target_node,
            attack_times,
            n_perturbations,
            original_label_item,
            final_label,
            initial_confidence,
            final_confidence,
        )

    def rerank_attack(
        self,
        ori_features,
        ori_adj,
        target_node,
        n_perturbations,
        strategy,
        *,
        target_label=0,
        res_gt=None,
        verbose=False,
        **kwargs,
    ):
        """Attack using reranking heuristics to avoid label flips.

        Extends :meth:`attack` with probability-derivative reranking that downweights edge
        candidates likely to invert the predicted label during each perturbation step.

        Parameters
        ----------
        ori_features : torch.Tensor
            Node feature matrix of shape ``(n_nodes, n_features)``.
        ori_adj : torch.Tensor
            Dense adjacency matrix of shape ``(n_nodes, n_nodes)``.
        target_node : int
            Index of the node to attack.
        n_perturbations : int
            Maximum number of edge flips to apply.
        strategy : str
            Strategy keyword (reranking currently targets underconfidence).
        res_gt : torch.Tensor, optional
            Ground-truth labels used when evaluating calibration objectives (required).
        target_label : int, optional
            Placeholder to mirror the baseline signature (not used).
        verbose : bool, optional
            If ``True``, emit per-iteration diagnostics (default: ``False``).
        **kwargs : dict, optional
            Optional trackers such as ``n_perturb`` and ``best_conf`` lists for logging.

        Raises
        ------
        ValueError
            If ``res_gt`` is not provided.

        Returns
        -------
        None
            The best perturbed adjacency is stored on ``self.modified_adj``.
        """
        if strategy != 'under':
            raise ValueError(f"rerank_attack only supports 'under' strategy, got '{strategy}'")

        display_strategy = strategy

        # Get initial predictions and setup
        original_output = self.surrogate(ori_features, ori_adj).detach()[[target_node]]

        if res_gt is None:
            raise ValueError("res_gt must be provided for Calib_FGA attacks")

        res_gt = res_gt[[target_node]]
        original_label = original_output.argmax(1)
        original_label_item = int(original_label.item())

        # Use KL divergence with uniform distribution as loss criterion
        criterion = kl_divergence_with_uniform
        baseline_probs = F.softmax(original_output, dim=1)
        best_confidence = baseline_probs[0, original_label_item].item()
        initial_confidence = best_confidence

        self._log_attack_header(
            display_strategy,
            target_node,
            n_perturbations,
            original_label_item,
            initial_confidence,
        )

        # Initialize adjacency matrices
        modified_adj = ori_adj.detach().clone().to(self.device)
        modified_features = ori_features.detach().clone()
        best_adj = modified_adj.detach().clone()
        self.surrogate.eval()

        # Main attack loop with reranking
        modified_adj.requires_grad = True
        attack_times = 0

        table_rows = [] if verbose else None

        for i in range(n_perturbations):
            # Forward pass and loss computation
            output = self.surrogate(modified_features, modified_adj)[[target_node]]
            current_label = output.argmax(dim=1)
            prev_confidence = F.softmax(output.detach(), dim=1)[0, current_label].item()

            loss = criterion(output, current_label)
            grad = torch.autograd.grad(loss, modified_adj, retain_graph=True)[0]

            # Apply symmetry balancing
            # For unconnected edges (A=0): use positive gradients
            # For connected edges (A=1): use negative gradients
            delta_A = (-2*modified_adj[target_node] + 1)
            grad = (grad[target_node] + grad[:, target_node]) * delta_A

            # Compute label-flip prediction using probability derivatives
            probabilities = F.softmax(output, dim=1)
            p_max, p_smax = torch.topk(probabilities, 2, dim=1)[0][0]

            # Calculate gradients of top-2 probabilities w.r.t. adjacency matrix
            div_pmax = torch.autograd.grad(p_max, modified_adj, retain_graph=True)[0]
            div_psmax = torch.autograd.grad(p_smax, modified_adj, retain_graph=True)[0]

            # Predict label flip: if p_max + Δp_max < p_smax + Δp_smax after perturbation
            condition_matrix = p_max + div_pmax[target_node] * delta_A - p_smax - div_psmax[target_node] * delta_A
            label_flip_flag_matrix = torch.where(condition_matrix > 0, torch.tensor(1), torch.tensor(-1))

            # Apply reranking: downweight gradients for edges likely to cause label flips
            grad = grad * label_flip_flag_matrix
            grad[target_node] = -10  # Prevent self-loops
            grad_argmax = torch.argmax(grad)

            # Apply edge perturbation
            value = -2*modified_adj[target_node][grad_argmax] + 1
            modified_adj.data[target_node][grad_argmax] += value
            modified_adj.data[grad_argmax][target_node] += value

            # Verify label preservation
            with torch.no_grad():
                new_output = self.surrogate(modified_features, modified_adj)[[target_node]]
            new_label = new_output.argmax(dim=1)

            if new_label != original_label:
                print(
                    f"[{display_strategy.upper()}][Node {target_node}] "
                    f"Early stop at step {i + 1}: label flipped to {int(new_label.item())}"
                )
                break

            # Update best configuration (always seeks underconfidence)
            current_label = new_label
            predicted_label = int(current_label.item())
            current_confidence = F.softmax(new_output, dim=1)[0, predicted_label].item()
            attack_times += 1
            if current_confidence <= best_confidence:
                best_confidence = current_confidence
                best_adj = modified_adj.detach().clone()

            # Print progress information
            if verbose:
                action = "Added" if value > 0 else "Removed"
                self._log_iteration(
                    display_strategy,
                    target_node,
                    i + 1,
                    n_perturbations,
                    action,
                    int(grad_argmax.item()),
                    prev_confidence,
                    current_confidence,
                    initial_confidence,
                    predicted_label,
                    loss.item(),
                    log_rows=table_rows,
                )

        # Store results and finalize
        kwargs["n_perturb"].append(attack_times)
        kwargs["best_conf"].append(best_confidence)

        modified_adj = modified_adj.detach()
        self.check_adj(modified_adj)
        self.modified_adj = best_adj

        if verbose and table_rows:
            headers = ["Step", "Action", "Pred", "Conf", "Δ step", "Δ total", "Loss"]
            print(tabulate(table_rows, headers=headers, tablefmt="grid"))

        with torch.no_grad():
            final_output = self.surrogate(ori_features, self.modified_adj).detach()[[target_node]]
            final_label = int(final_output.argmax(dim=1).item())
            final_confidence = float(
                F.softmax(final_output, dim=1)[0, final_label].item()
            )

        self._log_attack_summary(
            display_strategy,
            target_node,
            attack_times,
            n_perturbations,
            original_label_item,
            final_label,
            initial_confidence,
            final_confidence,
        )

    def rerank_hybridloss_attack(
        self,
        ori_features,
        ori_adj,
        target_node,
        n_perturbations,
        strategy,
        *,
        target_label=0,
        res_gt=None,
        verbose=False,
        **kwargs,
    ):
        """Rerank while adaptively switching between calibration and restoration losses.

        Combines reranking with a hybrid objective: when the current prediction matches the
        baseline label, the calibration loss (KL to uniform) drives the update; once the prediction
        drifts, a classification loss nudges the surrogate back to the original label before
        continuing the calibration push.

        Parameters
        ----------
        ori_features : torch.Tensor
            Node feature matrix of shape ``(n_nodes, n_features)``.
        ori_adj : torch.Tensor
            Dense adjacency matrix of shape ``(n_nodes, n_nodes)``.
        target_node : int
            Index of the node to attack.
        n_perturbations : int
            Maximum number of edge flips to apply.
        strategy : str
            Strategy keyword (hybrid loss currently targets underconfidence).
        res_gt : torch.Tensor, optional
            Ground-truth labels used when evaluating calibration objectives (required).
        target_label : int, optional
            Placeholder to mirror the baseline signature (not used).
        verbose : bool, optional
            If ``True``, emit per-iteration diagnostics (default: ``False``).
        **kwargs : dict, optional
            Optional trackers such as ``n_perturb`` and ``best_conf`` lists for logging.

        Raises
        ------
        ValueError
            If ``res_gt`` is not provided.

        Returns
        -------
        None
            The best perturbed adjacency is stored on ``self.modified_adj``.
        """
        if strategy != 'under':
            raise ValueError(f"rerank_hybridloss_attack only supports 'under' strategy, got '{strategy}'")

        display_strategy = strategy

        # Get initial predictions and setup
        original_output = self.surrogate(ori_features, ori_adj).detach()[[target_node]]

        if res_gt is None:
            raise ValueError("res_gt must be provided for Calib_FGA attacks")

        res_gt = res_gt[[target_node]]
        original_label = original_output.argmax(1)
        original_label_item = int(original_label.item())

        # Define hybrid loss functions
        calib_criterion = kl_divergence_with_uniform  # For calibration when label unchanged
        class_criterion = lambda output, label: -F.nll_loss(output, label)  # For label restoration
        baseline_probs = F.softmax(original_output, dim=1)
        best_confidence = baseline_probs[0, original_label_item].item()
        initial_confidence = best_confidence

        self._log_attack_header(
            display_strategy,
            target_node,
            n_perturbations,
            original_label_item,
            initial_confidence,
        )

        # Initialize adjacency matrices
        modified_adj = ori_adj.detach().clone().to(self.device)
        modified_features = ori_features.detach().clone()
        best_adj = modified_adj.detach().clone()
        self.surrogate.eval()

        # Main attack loop with hybrid loss
        modified_adj.requires_grad = True
        attack_times = 0

        table_rows = [] if verbose else None

        for i in range(n_perturbations):
            # Forward pass and adaptive loss selection
            output = self.surrogate(modified_features, modified_adj)[[target_node]]
            current_label = output.argmax(dim=1)
            prev_confidence = F.softmax(output.detach(), dim=1)[0, current_label].item()

            # Adaptive loss switching based on current prediction
            if current_label == original_label:
                # Use calibration loss when prediction is correct
                loss = calib_criterion(output, current_label)
                loss_mode = "calib"
            else:
                # Use classification loss to restore original label
                loss = class_criterion(output, original_label)
                loss_mode = "restore"

            # Compute gradients w.r.t. adjacency matrix
            grad = torch.autograd.grad(loss, modified_adj, retain_graph=True)[0]

            # Apply symmetry balancing
            delta_A = (-2*modified_adj[target_node] + 1)
            grad = (grad[target_node] + grad[:, target_node]) * delta_A

            # Apply reranking only when using calibration loss (label unchanged)
            if current_label == original_label:
                # Compute label-flip prediction using probability derivatives
                probabilities = F.softmax(output, dim=1)
                p_max, p_smax = torch.topk(probabilities, 2, dim=1)[0][0]

                # Calculate gradients of top-2 probabilities
                div_pmax = torch.autograd.grad(p_max, modified_adj, retain_graph=True)[0]
                div_psmax = torch.autograd.grad(p_smax, modified_adj, retain_graph=True)[0]

                # Predict label flip and apply reranking
                condition_matrix = p_max + div_pmax[target_node] * delta_A - p_smax - div_psmax[target_node] * delta_A
                label_flip_flag_matrix = torch.where(condition_matrix > 0, torch.tensor(1), torch.tensor(-1))
                grad = grad * label_flip_flag_matrix

            grad[target_node] = -10  # Prevent self-loops
            grad_argmax = torch.argmax(grad)

            # Apply edge perturbation
            value = -2*modified_adj[target_node][grad_argmax] + 1
            modified_adj.data[target_node][grad_argmax] += value
            modified_adj.data[grad_argmax][target_node] += value

            # Verify label preservation (final check)
            with torch.no_grad():
                new_output = self.surrogate(modified_features, modified_adj)[[target_node]]
            new_label = new_output.argmax(dim=1)

            if new_label != original_label:
                print(
                    f"[{display_strategy.upper()}][Node {target_node}] "
                    f"Early stop at step {i + 1}: label flipped to {int(new_label.item())}"
                )
                break

            # Update best configuration (always seeks underconfidence)
            current_label = new_label
            predicted_label = int(current_label.item())
            current_confidence = F.softmax(new_output, dim=1)[0, predicted_label].item()
            attack_times += 1
            if current_confidence <= best_confidence:
                best_confidence = current_confidence
                best_adj = modified_adj.detach().clone()

            # Print progress information
            if verbose:
                action = "Added" if value > 0 else "Removed"
                action = f"{action} [{loss_mode}]"
                self._log_iteration(
                    display_strategy,
                    target_node,
                    i + 1,
                    n_perturbations,
                    action,
                    int(grad_argmax.item()),
                    prev_confidence,
                    current_confidence,
                    initial_confidence,
                    predicted_label,
                    loss.item(),
                    log_rows=table_rows,
                )

        # Store results and finalize
        kwargs["n_perturb"].append(attack_times)
        kwargs["best_conf"].append(best_confidence)

        modified_adj = modified_adj.detach()
        self.check_adj(modified_adj)
        self.modified_adj = best_adj

        if verbose and table_rows:
            headers = ["Step", "Action", "Pred", "Conf", "Δ step", "Δ total", "Loss"]
            print(tabulate(table_rows, headers=headers, tablefmt="grid"))

        with torch.no_grad():
            final_output = self.surrogate(ori_features, self.modified_adj).detach()[[target_node]]
            final_label = int(final_output.argmax(dim=1).item())
            final_confidence = float(
                F.softmax(final_output, dim=1)[0, final_label].item()
            )

        self._log_attack_summary(
            display_strategy,
            target_node,
            attack_times,
            n_perturbations,
            original_label_item,
            final_label,
            initial_confidence,
            final_confidence,
        )

    def flip_beam_hybridloss_attack(
        self,
        ori_features,
        ori_adj,
        target_node,
        n_perturbations,
        strategy,
        *,
        target_label=0,
        res_gt=None,
        verbose=False,
        **kwargs,
    ):
        """Apply beam-search hybrid loss with reranking safeguards.

        Represents the full Calib-FGA framework: beam search explores multiple graph candidates,
        reranking discourages label flips, and the hybrid loss toggles between calibration and
        restoration to keep predictions stable while altering confidence.

        Parameters
        ----------
        ori_features : torch.Tensor
            Node feature matrix of shape ``(n_nodes, n_features)``.
        ori_adj : torch.Tensor
            Dense adjacency matrix of shape ``(n_nodes, n_nodes)``.
        target_node : int
            Index of the node to attack.
        n_perturbations : int
            Maximum number of edge flips to apply.
        strategy : str
            Strategy keyword (beam search currently targets underconfidence).
        res_gt : torch.Tensor, optional
            Ground-truth labels used when evaluating calibration objectives (required).
        target_label : int, optional
            Placeholder to mirror the baseline signature (not used).
        verbose : bool, optional
            If ``True``, emit per-iteration diagnostics including a tabulated summary (default: ``False``).
        **kwargs : dict, optional
            Supports ``beam_width`` (default ``3``), tracking lists (``n_perturb``, ``best_conf``), and
            optional ``runtime`` logging.

        Raises
        ------
        ValueError
            If ``res_gt`` is not provided.

        Returns
        -------
        None
            The best perturbed adjacency is stored on ``self.modified_adj``.
        """
        if strategy != 'under':
            raise ValueError(f"flip_beam_hybridloss_attack only supports 'under' strategy, got '{strategy}'")

        display_strategy = strategy

        # Get initial predictions and setup
        original_output = self.surrogate(ori_features, ori_adj).detach()[[target_node]]

        if res_gt is None:
            raise ValueError("res_gt must be provided for Calib_FGA attacks")

        res_gt = res_gt[[target_node]]
        original_label = original_output.argmax(1)
        original_label_item = int(original_label.item())

        # Define hybrid loss functions
        calib_criterion = kl_divergence_with_uniform  # For calibration when label unchanged
        class_criterion = lambda output, label: -F.nll_loss(output, label)  # For label restoration

        # Initialize adjacency matrices
        modified_adj = ori_adj.detach().clone().to(self.device)
        modified_features = ori_features.detach().clone().to(self.device)
        self.surrogate.eval()

        # Initialize beam search parameters
        beam_width = kwargs.get('beam_width', 3)
        beam = PriorityQueue()

        # Initialize beam with original state
        baseline_probs = F.softmax(original_output, dim=1)
        initial_confidence = baseline_probs[0, original_label_item].item()
        self._log_attack_header(
            display_strategy,
            target_node,
            n_perturbations,
            original_label_item,
            initial_confidence,
        )
        beam.put((initial_confidence, 0, modified_adj.clone()))
        best_adj = modified_adj.clone()
        best_confidence = initial_confidence
        attack_times = 0

        # Track progress for tabulated output
        table_data = []
        start_time = time.time()

        # Main beam search loop
        for iteration in range(n_perturbations):
            next_beam = PriorityQueue()

            # Explore all current beam candidates
            for _ in range(beam_width):
                if beam.empty():
                    break

                _, current_perturbations, current_adj = beam.get()

                if current_perturbations >= n_perturbations:
                    continue

                # Prepare current adjacency matrix for gradient computation
                current_adj_leaf = current_adj.clone().detach()
                current_adj_leaf.requires_grad = True

                # Forward pass and adaptive loss selection
                output = self.surrogate(modified_features, current_adj_leaf)[[target_node]]
                current_label = output.argmax(dim=1)

                # Adaptive loss switching
                if current_label == original_label:
                    loss = calib_criterion(output, current_label)
                else:
                    loss = class_criterion(output, original_label)

                grad = torch.autograd.grad(loss, current_adj_leaf, retain_graph=True)[0]
                
                # Apply symmetry balancing
                delta_A = (-2*current_adj_leaf[target_node] + 1)
                grad = (grad[target_node] + grad[:, target_node]) * delta_A

                # Apply reranking only when using calibration loss (label unchanged)
                if current_label == original_label:
                    # Compute label-flip prediction using probability derivatives
                    probabilities = F.softmax(output, dim=1)
                    p_max, p_smax = torch.topk(probabilities, 2, dim=1)[0][0]

                    div_pmax = torch.autograd.grad(p_max, current_adj_leaf, retain_graph=True)[0]
                    div_psmax = torch.autograd.grad(p_smax, current_adj_leaf, retain_graph=True)[0]
                    
                    # Predict label flip and apply reranking
                    condition_matrix = p_max + div_pmax[target_node] * delta_A - p_smax - div_psmax[target_node] * delta_A
                    label_flip_flag_matrix = torch.where(condition_matrix > 0, torch.tensor(1), torch.tensor(-1))
                    grad = grad * label_flip_flag_matrix

                grad[target_node] = -10  # Prevent self-loops
                grad_argmax = torch.argmax(grad)
                
                # Apply edge perturbation
                value = -2*current_adj_leaf[target_node][grad_argmax] + 1
                new_adj = current_adj_leaf.clone().detach()
                new_adj.data[target_node][grad_argmax] += value
                new_adj.data[grad_argmax][target_node] += value

                # Evaluate new configuration
                with torch.no_grad():
                    new_output = self.surrogate(modified_features, new_adj)[[target_node]]
                new_label = new_output.argmax(dim=1)
                new_confidence = F.softmax(new_output, dim=1)[0, new_label].item()

                # Add to next beam
                next_beam.put((new_confidence, current_perturbations + 1, new_adj))

                # Update global best if label preserved and confidence improved
                if new_label == original_label and new_confidence < best_confidence:
                    best_confidence = new_confidence
                    best_adj = new_adj.clone()
                    attack_times = current_perturbations + 1

            # Update beam for next iteration
            beam = next_beam

            # Record progress data
            table_data.append([
                iteration + 1,
                f"{initial_confidence:.4f}",
                f"{best_confidence:.4f}",
                attack_times
            ])

        # Finalize and report results
        end_time = time.time()
        elapsed_time = end_time - start_time

        if verbose:
            headers = ["Iteration", "Initial Confidence", "Best Confidence", "Perturbations"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print(f"Attack completed in {elapsed_time:.4f} seconds")

        # Store results
        kwargs["n_perturb"].append(attack_times)
        kwargs["best_conf"].append(best_confidence)

        if "runtime" in kwargs:
            kwargs["runtime"].append(elapsed_time)

        self.check_adj(best_adj)
        self.modified_adj = best_adj

        with torch.no_grad():
            final_output = self.surrogate(ori_features, self.modified_adj).detach()[[target_node]]
            final_label = int(final_output.argmax(dim=1).item())
            final_confidence = float(
                F.softmax(final_output, dim=1)[0, final_label].item()
            )
        if final_label != original_label_item:
            raise ValueError("Final label does not match original label!")

        self._log_attack_summary(
            display_strategy,
            target_node,
            attack_times,
            n_perturbations,
            original_label_item,
            final_label,
            initial_confidence,
            final_confidence,
        )
