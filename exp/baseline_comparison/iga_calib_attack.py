#!/usr/bin/env python3
"""
IGA (Integrated Gradients Attack) for GNN Calibration

This script demonstrates the Integrated Gradients Attack (IGA) on calibrated GNNs:
1. Training a base GNN model
2. Calibrating it with various calibration methods (CaGCN, ETS, GETS, MS, TS, VS, WATS, SimCalib)
3. Attacking the calibrated model using IGA, which uses integrated gradients to identify
   the most important edges to perturb for manipulating prediction confidence
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from datetime import datetime
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CitationFull

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calib_attack.calib_iga import Calib_IGA
from src.gnn.model import CompatibleGCN
from calibration.TS import TemperatureScaling
from calibration.VS import VectorScaling
from calibration.MS import MatrixScaling
from calibration.WATS import WATS
from calibration.CaGCN import CaGCN
from calibration.SimCalib import SimCalib
from calibration.DCGC import DCGC
from calibration.ETS import ETS
from calibration.GETS import GETSCalibrator as GETS
from calibration.GATS import GATSCalibrator as GATS
from utils.ece import calculate_average_ece
from calibration.utils import accuracy


def load_dataset(dataset_name):
    """Load specified dataset and return tensors for training and attacks."""
    dataset_name_lower = dataset_name.lower()

    # Load dataset based on name
    if dataset_name_lower in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root="./data", name=dataset_name.capitalize())
    elif dataset_name_lower in ['photo', 'computers']:
        dataset = Amazon(root="./data", name=dataset_name.capitalize())
    elif dataset_name_lower in ['cs', 'physics']:
        dataset = Coauthor(root="./data", name=dataset_name.upper())
    elif dataset_name_lower == 'dblp':
        dataset = CitationFull(root="./data", name='DBLP')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

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

    return data, features, adj, labels, dataset.num_features, dataset.num_classes, edge_index


def train_base_model(model, data, features, adj, labels, epochs=200):
    """Train the base GNN model."""
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


def save_attack_log(args, attack_results, base_metrics, calibrated_metrics, experiment_config, log_dir="./logs"):
    """Save comprehensive IGA attack results and experiment information to log files."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare comprehensive log data
    log_data = {
        "experiment_info": {
            "timestamp": timestamp,
            "attack_type": "IGA (Integrated Gradients Attack)",
            "calibration_method": f"{args.calibration_method}",
            "dataset": args.dataset,
            "config": experiment_config
        },
        "model_metrics": {
            "base_model": {
                "accuracy": base_metrics["accuracy"],
                "ece": base_metrics["ece"],
                "avg_confidence": base_metrics["avg_confidence"]
            },
            "calibrated_model": {
                "accuracy": calibrated_metrics["accuracy"],
                "ece": calibrated_metrics["ece"],
                "avg_confidence": calibrated_metrics["avg_confidence"]
            }
        },
        "attack_summary": {
            "total_nodes_attacked": len(attack_results),
            "strategies": list(set([r['strategy'] for r in attack_results])),
        },
        "detailed_results": attack_results
    }

    # Save JSON log
    json_file = os.path.join(log_dir, f"iga_attack_log_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    # Save human-readable summary
    summary_file = os.path.join(log_dir, f"iga_attack_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("IGA (Integrated Gradients Attack) EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Attack Type: IGA (Integrated Gradients Attack)\n")
        f.write(f"Calibration Method: {args.calibration_method}\n")
        f.write(f"Dataset: {args.dataset}\n\n")

        f.write("MODEL PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Base Model:\n")
        f.write(f"  Accuracy: {base_metrics['accuracy']:.4f}\n")
        f.write(f"  ECE: {base_metrics['ece']:.4f}\n")
        f.write(f"  Avg Confidence: {base_metrics['avg_confidence']:.4f}\n\n")

        f.write(f"Calibrated Model:\n")
        f.write(f"  Accuracy: {calibrated_metrics['accuracy']:.4f}\n")
        f.write(f"  ECE: {calibrated_metrics['ece']:.4f}\n")
        f.write(f"  Avg Confidence: {calibrated_metrics['avg_confidence']:.4f}\n")
        if 'temperature' in calibrated_metrics:
            f.write(f"  Temperature: {calibrated_metrics['temperature']:.4f}\n")
        f.write("\n")

        f.write("IGA ATTACK RESULTS:\n")
        f.write("-" * 40 + "\n")
        for strategy in ['under', 'over']:
            strategy_results = [r for r in attack_results if r['strategy'] == strategy]
            if strategy_results:
                label_preserved = [r for r in strategy_results if r['label_preserved']]
                if strategy == 'under':
                    successful = [r for r in label_preserved if r['conf_change'] < 0]
                else:
                    successful = [r for r in label_preserved if r['conf_change'] > 0]

                success_rate = len(successful) / len(strategy_results) if strategy_results else 0
                avg_conf_change = np.mean([r['conf_change'] for r in successful]) if successful else 0
                avg_perturbations = np.mean([r['perturbations'] for r in strategy_results])

                f.write(f"\n{strategy.upper()} Strategy:\n")
                f.write(f"  Nodes attacked: {len(strategy_results)}\n")
                f.write(f"  Successful attacks: {len(successful)}\n")
                f.write(f"  Success rate: {success_rate:.2%}\n")
                f.write(f"  Avg confidence change: {avg_conf_change:+.4f}\n")
                f.write(f"  Avg edge perturbations: {avg_perturbations:.1f}\n")

    print(f"\nIGA attack logs saved:")
    print(f"  JSON log: {json_file}")
    print(f"  Summary: {summary_file}")

    return json_file, summary_file


def run_calib_iga_attacks(attack_model, calibrated_model, features, adj, labels, target_nodes,
                         n_perturbations=10, strategies=['under', 'over'], integration_steps=10):
    """
    Run IGA (Integrated Gradients Attack) with different strategies on specified nodes.

    IGA uses integrated gradients to identify the most important edges to perturb
    for manipulating prediction confidence while preserving the predicted label.
    """
    print(f"\nRunning IGA (Integrated Gradients Attack) on {len(target_nodes)} nodes...")

    all_results = {}

    for strategy in strategies:
        print(f"\n{'='*20} STRATEGY: {strategy.upper()} {'='*20}")

        strategy_results = []
        successful_attacks = 0

        for i, node in enumerate(target_nodes):
            if (i + 1) % 5 == 0:
                print(f"Attacking node {node} ({i+1}/{len(target_nodes)})...")

            # Get original prediction and confidence before IGA attack
            with torch.no_grad():
                original_logits = calibrated_model(features, adj)
                if hasattr(original_logits, 'exp'):
                    original_probs = original_logits.exp()
                else:
                    original_probs = F.softmax(original_logits, dim=1)
                original_pred = original_probs[node].argmax().item()
                original_conf = original_probs[node].max().item()

            # Run IGA attack: uses integrated gradients to rank edge importance
            attack = Calib_IGA(attack_model, device=features.device)

            n_perturb = []  # Track number of edge perturbations
            best_conf = []  # Track best confidence achieved

            attack.attack(
                ori_features=features,
                ori_adj=adj,
                target_node=node,
                n_perturbations=n_perturbations,
                strategy=strategy,  # 'under' for underconfidence, 'over' for overconfidence
                res_gt=labels,
                steps=integration_steps,  # Number of integration steps for computing gradients
                verbose=True,
                n_perturb=n_perturb,
                best_conf=best_conf
            )

            # Evaluate model after IGA attack (with perturbed edges)
            with torch.no_grad():
                attacked_logits = calibrated_model(features, attack.modified_adj)
                if hasattr(attacked_logits, 'exp'):
                    attacked_probs = attacked_logits.exp()
                else:
                    attacked_probs = F.softmax(attacked_logits, dim=1)
                attacked_pred = attacked_probs[node].argmax().item()
                attacked_conf = attacked_probs[node].max().item()

            # Store IGA attack results
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
                if strategy == 'under' and result['conf_change'] < 0:
                    successful_attacks += 1
                elif strategy == 'over' and result['conf_change'] > 0:
                    successful_attacks += 1

        all_results[strategy] = strategy_results

        # Print strategy summary
        print(f"\n{strategy.upper()} IGA Attack Summary:")
        print(f"Successful IGA attacks: {successful_attacks}/{len(target_nodes)} ({successful_attacks/len(target_nodes)*100:.1f}%)")

        successful_results = [r for r in strategy_results if r['label_preserved']]
        if successful_results:
            avg_conf_change = np.mean([r['conf_change'] for r in successful_results])
            avg_perturbations = np.mean([r['perturbations'] for r in strategy_results])
            print(f"Average confidence change: {avg_conf_change:+.4f}")
            print(f"Average edge perturbations used: {avg_perturbations:.1f}")

        # Show some example IGA results
        print("\nExample IGA attack results:")
        for result in strategy_results[:3]:
            status = "SUCCESS" if result['label_preserved'] else "FAILED"
            print(f"Node {result['node']}: {result['original_conf']:.4f} → {result['attacked_conf']:.4f} "
                  f"({result['conf_change']:+.4f}) [{status}]")

    return all_results


def parse_arguments():
    """Parse command line arguments for IGA attack configuration."""
    parser = argparse.ArgumentParser(description='IGA (Integrated Gradients Attack) for Calibrated GNNs')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers', 'CS', 'Physics', 'DBLP'],
                        help='Dataset to use (default: Cora)')
    parser.add_argument('--calibration-method', '--calib-method',
                        choices=['CaGCN', 'DCGC', 'ETS', 'GATS', 'GETS', 'MS', 'TS', 'VS', 'WATS', 'SimCalib'],
                        default='TS',
                        help='Calibration method to use (default: TS)')
    parser.add_argument('--budget', type=int, default=8,
                        help='Number of edge perturbations allowed per node (default: 8)')

    # IGA attack configuration arguments
    parser.add_argument('--attack-nodes', type=int, default=20,
                        help='Number of nodes to attack with IGA (default: 20)')
    parser.add_argument('--strategies', nargs='+', default=['under', 'over'],
                        choices=['under', 'over'],
                        help='Attack strategies to use (default: under over)')
    parser.add_argument('--integration-steps', type=int, default=10,
                        help='Number of integration steps for IGA (default: 10)')

    # GETS-specific arguments
    parser.add_argument('--gets-experts', type=int, default=3,
                        help='Number of experts for GETS (default: 3)')
    parser.add_argument('--gets-backbone', choices=['gcn', 'gat', 'gin'], default='gcn',
                        help='Backbone architecture for GETS experts (default: gcn)')
    parser.add_argument('--gets-hidden-dim', type=int, default=32,
                        help='Hidden dimension for GETS experts (default: 32)')

    return parser.parse_args()


def get_calibration_model(method, base_model, features, labels, adj, val_mask, num_classes, edge_index=None, args=None):
    """Initialize and return the appropriate calibration model based on the method."""
    device = features.device

    if method == 'TS':
        return TemperatureScaling(base_model, features, labels, adj, val_mask)
    elif method == 'VS':
        return VectorScaling(base_model, features, labels, adj, val_mask)
    elif method == 'MS':
        return MatrixScaling(base_model, features, labels, adj, val_mask)
    elif method == 'WATS':
        return WATS(base_model, features, labels, adj, val_mask)
    elif method == 'CaGCN':
        return CaGCN(base_model, features, labels, adj, val_mask)
    elif method == 'DCGC':
        return DCGC(base_model, features, labels, adj, val_mask)
    elif method == 'SimCalib':
        return SimCalib(base_model, features, labels, adj, val_mask)
    elif method == 'ETS':
        model = ETS(base_model, features, labels, adj, val_mask)
        model.fit([None, val_mask, None])
        return model
    elif method == 'GATS':
        return GATS(base_model, features, labels, adj, val_mask)
    elif method == 'GETS':
        conf = {
            'lr': 0.01,
            'max_epoch': 100,
            'patience': 10,
            'weight_decay': 5e-4
        }
        num_experts = args.gets_experts if args else 3
        backbone = args.gets_backbone if args else 'gcn'
        hidden_dim = args.gets_hidden_dim if args else 32

        model = GETS(
            base_model=base_model,
            num_classes=num_classes,
            device=device,
            conf=conf,
            num_experts=num_experts,
            expert_select=min(2, num_experts),
            hidden_dim=hidden_dim,
            dropout_rate=0.1,
            num_layers=2,
            feature_hidden_dim=max(16, hidden_dim // 2),
            degree_hidden_dim=8,
            noisy_gating=True,
            loss_coef=1e-2,
            backbone=backbone
        )
        model.fit(adj, features, labels, [None, val_mask, None])
        return model
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def compare_attack_strategies(attack_results):
    """Compare the effectiveness of different IGA attack strategies."""
    print("\n" + "="*60)
    print("IGA ATTACK STRATEGY COMPARISON")
    print("="*60)

    for strategy, results in attack_results.items():
        label_preserved = [r for r in results if r['label_preserved']]

        if strategy == 'under':
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


def analyze_integrated_gradients_effectiveness(attack_results):
    """Analyze the effectiveness of integrated gradients approach."""
    print("\n" + "="*60)
    print("INTEGRATED GRADIENTS ANALYSIS")
    print("="*60)

    all_results = []
    for strategy_results in attack_results.values():
        all_results.extend(strategy_results)

    # Analysis of perturbation efficiency
    successful_attacks = [r for r in all_results if r['label_preserved']]
    if successful_attacks:
        perturbations_used = [r['perturbations'] for r in successful_attacks]
        conf_changes = [abs(r['conf_change']) for r in successful_attacks]

        print(f"Total successful attacks: {len(successful_attacks)}/{len(all_results)} ({len(successful_attacks)/len(all_results)*100:.1f}%)")
        print(f"Average perturbations per successful attack: {np.mean(perturbations_used):.1f} ± {np.std(perturbations_used):.1f}")
        print(f"Average confidence change magnitude: {np.mean(conf_changes):.4f} ± {np.std(conf_changes):.4f}")

        # Efficiency analysis
        efficiency_scores = [abs(r['conf_change']) / max(r['perturbations'], 1) for r in successful_attacks]
        print(f"Average confidence change per perturbation: {np.mean(efficiency_scores):.4f}")

    print(f"\nIntegrated Gradients characteristics:")
    print(f"- Uses principled gradient integration for edge importance ranking")
    print(f"- More computationally expensive than simple gradient methods")
    print(f"- Provides better attribution for edge perturbations")
    print(f"- Suitable for targeted calibration attacks")


def main():
    # Parse arguments
    args = parse_arguments()

    # Set device and seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"Using device: {device}")
    print(f"Using dataset: {args.dataset}")
    print(f"Using calibration method: {args.calibration_method}")

    # Load data
    data, features, adj, labels, num_features, num_classes, edge_index = load_dataset(args.dataset)
    print(f"Loaded {args.dataset}: {data.num_nodes} nodes, {num_classes} classes")

    # Move to device
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)

    # Step 1: Train base model
    base_model = CompatibleGCN(num_features, dataset_name=args.dataset.lower()).to(device)
    train_base_model(base_model, data, features, adj, labels)

    # Evaluate base model
    print("\n" + "="*50)
    print("EVALUATING BASE MODEL")
    print("="*50)
    base_acc, base_ece, base_conf = evaluate_model_calibration(
        base_model, features, adj, labels, data.test_mask, "Base Model"
    )

    # Step 2: Apply selected calibration method
    print("\n" + "="*50)
    print(f"APPLYING {args.calibration_method.upper()} CALIBRATION")
    print("="*50)
    calibrated_model = get_calibration_model(
        args.calibration_method, base_model, features, labels, adj, data.val_mask, num_classes, edge_index, args
    )

    # Evaluate calibrated model
    print("\nEvaluating calibrated model...")
    calib_acc, calib_ece, calib_conf = evaluate_model_calibration(
        calibrated_model, features, adj, labels, data.test_mask, f"{args.calibration_method} Calibrated Model"
    )

    print(f"\nCalibration improvement:")
    print(f"ECE change: {calib_ece - base_ece:+.4f}")

    # Prepare metrics for logging
    base_metrics = {
        "accuracy": base_acc,
        "ece": base_ece,
        "avg_confidence": base_conf
    }

    calibrated_metrics = {
        "accuracy": calib_acc,
        "ece": calib_ece,
        "avg_confidence": calib_conf,
    }

    # Step 3: Launch IGA attack on the calibrated model
    print("\n" + "="*50)
    print("LAUNCHING IGA (INTEGRATED GRADIENTS ATTACK) ON CALIBRATED MODEL")
    print("="*50)

    # Select test nodes for IGA attack
    available_test_nodes = torch.nonzero(data.test_mask).flatten()
    num_attack_nodes = min(args.attack_nodes, len(available_test_nodes))
    test_node_indices = available_test_nodes[:num_attack_nodes].cpu().numpy()

    print(f"Selected {num_attack_nodes} nodes for IGA attack (out of {len(available_test_nodes)} test nodes)")
    print(f"IGA configuration: {args.budget} edge perturbations per node, {args.integration_steps} integration steps")
    print(f"Attack strategies: {args.strategies}")

    # Experiment configuration for logging
    experiment_config = {
        "target_nodes": len(test_node_indices),
        "budget": args.budget,
        "strategies": args.strategies,
        "integration_steps": args.integration_steps,
        "attack_mechanism": "integrated_gradients_based",
        "base_model": "CompatibleGCN",
        "training_epochs": 200,
        "learning_rate": 0.01,
        "weight_decay": 5e-4
    }

    # Run IGA attacks with different strategies
    attack_results_dict = run_calib_iga_attacks(
        attack_model=calibrated_model,  # Use calibrated model for computing integrated gradients
        calibrated_model=calibrated_model,  # Attack the calibrated model
        features=features,
        adj=adj,
        labels=labels,
        target_nodes=test_node_indices,
        n_perturbations=args.budget,
        strategies=args.strategies,
        integration_steps=args.integration_steps
    )

    # Flatten results for logging
    attack_results = []
    for strategy_results in attack_results_dict.values():
        attack_results.extend(strategy_results)

    # Compare strategies
    compare_attack_strategies(attack_results_dict)

    # Analyze integrated gradients effectiveness
    analyze_integrated_gradients_effectiveness(attack_results_dict)

    # Save comprehensive IGA logs
    print("\n" + "="*50)
    print("SAVING IGA ATTACK LOGS")
    print("="*50)
    save_attack_log(args, attack_results, base_metrics, calibrated_metrics, experiment_config)

    # Overall analysis
    print("\n" + "="*60)
    print("OVERALL ANALYSIS")
    print("="*60)

    print(f"Dataset: {args.dataset}")
    print(f"Calibration method: {args.calibration_method}")
    print(f"Base model calibration (ECE): {base_ece:.4f}")
    print(f"{args.calibration_method} calibrated model (ECE): {calib_ece:.4f}")
    print(f"Calibration improvement: {base_ece - calib_ece:+.4f}")

    print(f"\nIGA attack characteristics:")
    print(f"- Uses integrated gradients for principled edge importance ranking")
    print(f"- More computationally intensive than gradient-based methods")
    print(f"- Provides better attribution and interpretability")
    print(f"- Effective for targeted calibration manipulation")

    # Find best performing strategy
    best_strategy = None
    best_success_rate = 0

    for strategy, results in attack_results_dict.items():
        label_preserved = [r for r in results if r['label_preserved']]
        if strategy == 'under':
            successful = [r for r in label_preserved if r['conf_change'] < 0]
        else:
            successful = [r for r in label_preserved if r['conf_change'] > 0]

        success_rate = len(successful) / len(results) if results else 0
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_strategy = strategy

    if best_strategy:
        print(f"\nMost effective IGA strategy: {best_strategy} ({best_success_rate*100:.1f}% success rate)")

    # Comparison with other methods
    print(f"\nComparison with other attack methods:")
    print(f"- IGA provides more principled edge selection than random attacks")
    print(f"- More computationally expensive than FGA but potentially more effective")
    print(f"- Better interpretability of attack decisions through integrated gradients")
    print(f"- Suitable for both underconfidence and overconfidence attacks")


if __name__ == "__main__":
    main()