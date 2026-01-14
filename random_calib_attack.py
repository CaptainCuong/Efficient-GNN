#!/usr/bin/env python3
"""
Temperature Scaling Calibration followed by Calib_Random Attack

This script demonstrates:
1. Training a base GNN model
2. Calibrating it with Temperature Scaling (TS)
3. Attacking the calibrated model using Calib_Random (random baseline attack)
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid

from calib_attack.calib_random import Calib_Random
from src.gnn.model import CompatibleGCN
from calib.TS import TemperatureScaling
from utils.ece import calculate_average_ece
from calib.utils import accuracy


def load_cora():
    """Load Cora dataset and return tensors for training and attacks."""
    dataset = Planetoid(root="./data", name="Cora")
    data = dataset[0]

    num_nodes = data.num_nodes
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    edge_index = data.edge_index
    adj[edge_index[0], edge_index[1]] = 1.0
    adj = adj + adj.t()  # Make symmetric
    adj = torch.clamp(adj, 0, 1)  # Ensure values are between 0 and 1
    adj.fill_diagonal_(1.0)  # Add self-loops
    features = data.x.float()
    labels = data.y

    return data, features, adj, labels, dataset.num_features, dataset.num_classes


def train_base_model(model, data, features, adj, labels, epochs=200):
    """Train the base GNN model on Cora."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("Training base model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adj)
        loss = F.cross_entropy(logits[data.train_mask], labels[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(features, adj).argmax(dim=1)
                val_acc = (pred[data.val_mask] == labels[data.val_mask]).float().mean().item()
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

    model.eval()
    print("Base model training completed.")


def evaluate_model_calibration(model, features, adj, labels, test_mask, model_name="Model"):
    """Evaluate calibration metrics for a model."""
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        if hasattr(logits, 'exp'):  # log_softmax output (from TS calibrated model)
            probs = logits.exp()
        else:  # raw logits
            probs = F.softmax(logits, dim=1)

        test_probs = probs[test_mask]
        test_labels = labels[test_mask]

        # Accuracy
        pred_labels = torch.argmax(test_probs, dim=1)
        acc = (pred_labels == test_labels).float().mean().item()

        # ECE
        num_classes = test_probs.shape[1]
        ece = calculate_average_ece(test_probs.cpu().numpy(), test_labels.cpu().numpy(), num_classes, logits=False)

        # Average confidence
        confidence = torch.max(test_probs, dim=1)[0]
        avg_confidence = confidence.mean().item()

        print(f"{model_name} - Accuracy: {acc:.4f} | ECE: {ece:.4f} | Avg Confidence: {avg_confidence:.4f}")
        return acc, ece, avg_confidence


def run_random_attacks(attack_model, calibrated_model, features, adj, labels, target_nodes,
                      n_perturbations=5, strategies=['under', 'over', 'under_kl']):
    """Run Calib_Random attacks with different strategies on specified nodes."""
    print(f"\nRunning Calib_Random attacks on {len(target_nodes)} nodes...")

    all_results = {}

    for strategy in strategies:
        print(f"\n{'='*20} STRATEGY: {strategy.upper()} {'='*20}")

        strategy_results = []
        successful_attacks = 0

        for i, node in enumerate(target_nodes):
            if (i + 1) % 5 == 0:
                print(f"Attacking node {node} ({i+1}/{len(target_nodes)})...")

            # Get original prediction and confidence
            with torch.no_grad():
                original_logits = calibrated_model(features, adj)
                if hasattr(original_logits, 'exp'):
                    original_probs = original_logits.exp()
                else:
                    original_probs = F.softmax(original_logits, dim=1)
                original_pred = original_probs[node].argmax().item()
                original_conf = original_probs[node].max().item()

            # Run attack
            attack = Calib_Random(attack_model, attack_structure=True, device=features.device)

            n_perturb = []
            best_conf = []

            attack.attack(
                ori_features=features,
                ori_adj=adj,
                target_node=node,
                n_perturbations=n_perturbations,
                strategy=strategy,
                max_trials=50,
                verbose=False,
                n_perturb=n_perturb,
                best_conf=best_conf
            )

            # Evaluate attacked model
            with torch.no_grad():
                attacked_logits = calibrated_model(features, attack.modified_adj)
                if hasattr(attacked_logits, 'exp'):
                    attacked_probs = attacked_logits.exp()
                else:
                    attacked_probs = F.softmax(attacked_logits, dim=1)
                attacked_pred = attacked_probs[node].argmax().item()
                attacked_conf = attacked_probs[node].max().item()

            # Store results
            result = {
                'node': node,
                'strategy': strategy,
                'original_pred': original_pred,
                'original_conf': original_conf,
                'attacked_pred': attacked_pred,
                'attacked_conf': attacked_conf,
                'conf_change': attacked_conf - original_conf,
                'perturbations': n_perturb[0] if n_perturb else 0,
                'label_preserved': original_pred == attacked_pred
            }
            strategy_results.append(result)

            # Count successful attacks based on strategy
            if result['label_preserved']:
                if strategy in ['under', 'under_kl'] and result['conf_change'] < 0:
                    successful_attacks += 1
                elif strategy == 'over' and result['conf_change'] > 0:
                    successful_attacks += 1

        all_results[strategy] = strategy_results

        # Print strategy summary
        print(f"\n{strategy.upper()} Attack Summary:")
        print(f"Successful attacks: {successful_attacks}/{len(target_nodes)} ({successful_attacks/len(target_nodes)*100:.1f}%)")

        successful_results = [r for r in strategy_results if r['label_preserved']]
        if successful_results:
            avg_conf_change = np.mean([r['conf_change'] for r in successful_results])
            avg_perturbations = np.mean([r['perturbations'] for r in strategy_results])
            print(f"Average confidence change: {avg_conf_change:+.4f}")
            print(f"Average perturbations used: {avg_perturbations:.1f}")

        # Show some example results
        print("\nExample results:")
        for result in strategy_results[:3]:
            status = "SUCCESS" if result['label_preserved'] else "FAILED"
            print(f"Node {result['node']}: {result['original_conf']:.4f} → {result['attacked_conf']:.4f} "
                  f"({result['conf_change']:+.4f}) [{status}]")

    return all_results


def compare_attack_strategies(attack_results):
    """Compare the effectiveness of different attack strategies."""
    print("\n" + "="*60)
    print("ATTACK STRATEGY COMPARISON")
    print("="*60)

    for strategy, results in attack_results.items():
        label_preserved = [r for r in results if r['label_preserved']]

        if strategy in ['under', 'under_kl']:
            successful = [r for r in label_preserved if r['conf_change'] < 0]
            objective = "confidence reduction"
        else:  # over
            successful = [r for r in label_preserved if r['conf_change'] > 0]
            objective = "confidence increase"

        success_rate = len(successful) / len(results) * 100

        if successful:
            avg_conf_change = np.mean([r['conf_change'] for r in successful])
            avg_perturbations = np.mean([r['perturbations'] for r in results])
            std_conf_change = np.std([r['conf_change'] for r in successful])
        else:
            avg_conf_change = 0
            avg_perturbations = 0
            std_conf_change = 0

        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Average {objective}: {avg_conf_change:+.4f} ± {std_conf_change:.4f}")
        print(f"  Average perturbations: {avg_perturbations:.1f}")


def main():
    # Set device and seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Using device: {device}")

    # Load data
    data, features, adj, labels, num_features, num_classes = load_cora()
    print(f"Loaded Cora: {data.num_nodes} nodes, {num_classes} classes")

    # Move to device
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)

    # Step 1: Train base model
    base_model = CompatibleGCN(num_features, dataset_name="cora").to(device)
    train_base_model(base_model, data, features, adj, labels)

    # Evaluate base model
    print("\n" + "="*50)
    print("EVALUATING BASE MODEL")
    print("="*50)
    base_acc, base_ece, base_conf = evaluate_model_calibration(
        base_model, features, adj, labels, data.test_mask, "Base Model"
    )

    # Step 2: Apply Temperature Scaling calibration
    print("\n" + "="*50)
    print("APPLYING TEMPERATURE SCALING CALIBRATION")
    print("="*50)
    calibrated_model = TemperatureScaling(base_model, features, labels, adj, data.val_mask)

    # Evaluate calibrated model
    print("\nEvaluating calibrated model...")
    calib_acc, calib_ece, calib_conf = evaluate_model_calibration(
        calibrated_model, features, adj, labels, data.test_mask, "TS Calibrated Model"
    )

    print(f"\nCalibration improvement:")
    print(f"ECE change: {calib_ece - base_ece:+.4f}")
    print(f"Temperature parameter: {calibrated_model.temperature.item():.4f}")

    # Step 3: Attack the calibrated model with random attacks
    print("\n" + "="*50)
    print("ATTACKING CALIBRATED MODEL WITH CALIB_RANDOM")
    print("="*50)

    # Select test nodes for attack (first 25 test nodes)
    test_node_indices = torch.nonzero(data.test_mask).flatten()[:25].cpu().numpy()

    # Run attacks with different strategies
    attack_results = run_random_attacks(
        attack_model=base_model,  # Use base model for gradient computation
        calibrated_model=calibrated_model,  # Attack the calibrated model
        features=features,
        adj=adj,
        labels=labels,
        target_nodes=test_node_indices,
        n_perturbations=8,
        strategies=['under', 'over', 'under_kl']
    )

    # Compare strategies
    compare_attack_strategies(attack_results)

    # Overall analysis
    print("\n" + "="*60)
    print("OVERALL ANALYSIS")
    print("="*60)

    print(f"Base model calibration (ECE): {base_ece:.4f}")
    print(f"TS calibrated model (ECE): {calib_ece:.4f}")
    print(f"Calibration improvement: {base_ece - calib_ece:+.4f}")

    print(f"\nRandom attack baseline effectiveness:")
    print(f"- Tests robustness against simple random perturbations")
    print(f"- No gradient information used (purely random)")
    print(f"- Useful for establishing attack difficulty baselines")

    # Find best performing strategy
    best_strategy = None
    best_success_rate = 0

    for strategy, results in attack_results.items():
        label_preserved = [r for r in results if r['label_preserved']]
        if strategy in ['under', 'under_kl']:
            successful = [r for r in label_preserved if r['conf_change'] < 0]
        else:
            successful = [r for r in label_preserved if r['conf_change'] > 0]

        success_rate = len(successful) / len(results)
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_strategy = strategy

    print(f"\nMost effective random strategy: {best_strategy} ({best_success_rate*100:.1f}% success rate)")


if __name__ == "__main__":
    main()