#!/usr/bin/env python3
"""
GNN Calibration Attack Demo

This script demonstrates:
1. Training a base GNN model
2. Calibrating it with various calibration methods (CaGCN, ETS, GETS, MS, TS, VS, WATS, SimCalib)
3. Attacking the calibrated model using flip_beam_hybridloss_attack from Calib_FGA
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from datetime import datetime
from torch_geometric.datasets import Planetoid

from calib_attack.calib_fga import Calib_FGA
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

    return data, features, adj, labels, dataset.num_features, dataset.num_classes, edge_index


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


def save_attack_log(args, attack_results, base_metrics, calibrated_metrics, experiment_config, log_dir="./logs"):
    """Save comprehensive attack results and experiment information to log files."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare comprehensive log data
    log_data = {
        "experiment_info": {
            "timestamp": timestamp,
            "attack_type": "flip_beam_hybridloss_attack",
            "calibration_method": f"{args.calibration_method}",
            "dataset": "Cora",
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
            "successful_attacks": len([r for r in attack_results if r['label_preserved'] and r['conf_change'] < 0]),
            "success_rate": len([r for r in attack_results if r['label_preserved'] and r['conf_change'] < 0]) / len(attack_results),
            "avg_confidence_reduction": np.mean([r['conf_change'] for r in attack_results if r['label_preserved'] and r['conf_change'] < 0]),
            "avg_perturbations": np.mean([r['perturbations'] for r in attack_results])
        },
        "detailed_results": attack_results
    }

    # Save JSON log
    json_file = os.path.join(log_dir, f"attack_log_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    # Save human-readable summary
    summary_file = os.path.join(log_dir, f"attack_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ATTACK EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Attack Type: flip_beam_hybridloss_attack\n")
        f.write(f"Calibration Method: Temperature Scaling\n")
        f.write(f"Dataset: Cora\n\n")

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

        f.write("ATTACK RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total nodes attacked: {len(attack_results)}\n")
        f.write(f"Successful attacks: {log_data['attack_summary']['successful_attacks']}\n")
        f.write(f"Success rate: {log_data['attack_summary']['success_rate']:.2%}\n")
        f.write(f"Avg confidence reduction: {log_data['attack_summary']['avg_confidence_reduction']:.4f}\n")
        f.write(f"Avg perturbations: {log_data['attack_summary']['avg_perturbations']:.1f}\n\n")

        f.write("SAMPLE RESULTS:\n")
        f.write("-" * 40 + "\n")
        for result in attack_results[:10]:
            status = "SUCCESS" if result['label_preserved'] and result['conf_change'] < 0 else "FAILED"
            f.write(f"Node {result['node']}: {result['original_conf']:.4f} → {result['attacked_conf']:.4f} "
                   f"({result['conf_change']:+.4f}) [{status}]\n")

    print(f"\nAttack logs saved:")
    print(f"  JSON log: {json_file}")
    print(f"  Summary: {summary_file}")

    return json_file, summary_file


def run_attack_on_nodes(attack_model, calibrated_model, features, adj, labels, target_nodes, budget=5):
    """Run flip_beam_hybridloss_attack on specified nodes."""
    print(f"\nRunning flip_beam_hybridloss_attack on {len(target_nodes)} nodes with budget {budget}...")

    attack_results = []
    successful_attacks = 0

    for i, node in enumerate(target_nodes):
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
        attack = Calib_FGA(attack_model, device=features.device)

        n_perturb = []
        best_conf = []

        attack.flip_beam_hybridloss_attack(
            ori_features=features,
            ori_adj=adj,
            target_node=node,
            n_perturbations=budget,
            strategy="under",
            res_gt=labels,
            verbose=False,
            beam_width=3,
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

        if original_pred != attacked_pred:
            raise ValueError("Attack failed: original and attacked predictions do not match.")

        # Store results
        result = {
            'node': int(node),
            'original_pred': int(original_pred),
            'original_conf': float(original_conf),
            'attacked_pred': int(attacked_pred),
            'attacked_conf': float(attacked_conf),
            'conf_change': float(attacked_conf - original_conf),
            'perturbations': int(n_perturb[0] if n_perturb else 0),
            'label_preserved': bool(original_pred == attacked_pred)
        }
        attack_results.append(result)

        if result['label_preserved'] and result['conf_change'] < 0:
            successful_attacks += 1

        print(f"  Original: pred={original_pred}, conf={original_conf:.4f}")
        print(f"  Attacked: pred={attacked_pred}, conf={attacked_conf:.4f}")
        print(f"  Change: {result['conf_change']:+.4f}, Perturbations: {result['perturbations']}")

    print(f"\nAttack Summary:")
    print(f"Successful attacks (label preserved + confidence reduced): {successful_attacks}/{len(target_nodes)}")

    return attack_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calibration Attack Demo')
    parser.add_argument('--calibration-method', '--calib-method',
                        choices=['CaGCN', 'DCGC', 'ETS', 'GATS', 'GETS', 'MS', 'TS', 'VS', 'WATS', 'SimCalib'],
                        default='TS',
                        help='Calibration method to use (default: TS)')
    parser.add_argument('--budget', type=int, default=5,
                        help='Number of perturbations allowed for attack (default: 5)')

    # Attack configuration arguments
    parser.add_argument('--attack-nodes', type=int, default=100,
                        help='Number of nodes to attack (default: 100)')

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
        # ETS requires different parameters
        model = ETS(base_model, features, labels, adj, val_mask)
        model.fit([None, val_mask, None])  # masks: [train, val, test]
        return model
    elif method == 'GATS':
        # GATS now generates edge_index internally from adjacency matrix
        return GATS(base_model, features, labels, adj, val_mask)
    elif method == 'GETS':
        # GETS requires different parameters - Enhanced configuration
        conf = {
            'lr': 0.01,
            'max_epoch': 100,
            'patience': 10,
            'weight_decay': 5e-4
        }
        # Use command line arguments if available
        num_experts = args.gets_experts if args else 3
        backbone = args.gets_backbone if args else 'gcn'
        hidden_dim = args.gets_hidden_dim if args else 32

        model = GETS(
            base_model=base_model,
            num_classes=num_classes,
            device=device,
            conf=conf,
            num_experts=num_experts,
            expert_select=min(2, num_experts),  # Ensure expert_select <= num_experts
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


def main():
    # Parse arguments
    args = parse_arguments()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using calibration method: {args.calibration_method}")

    # Load data
    data, features, adj, labels, num_features, num_classes, edge_index = load_cora()
    print(f"Loaded Cora: {data.num_nodes} nodes, {num_classes} classes")

    # Move to device
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
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

    # Step 3: Attack the calibrated model
    print("\n" + "="*50)
    print("ATTACKING CALIBRATED MODEL WITH FLIP_BEAM_HYBRIDLOSS_ATTACK")
    print("="*50)

    # Select test nodes for attack
    available_test_nodes = torch.nonzero(data.test_mask).flatten()
    num_attack_nodes = min(args.attack_nodes, len(available_test_nodes))
    test_node_indices = available_test_nodes[:num_attack_nodes].cpu().numpy()
    budget = args.budget

    print(f"Selected {num_attack_nodes} nodes for attack (out of {len(available_test_nodes)} test nodes)")

    # Experiment configuration for logging
    experiment_config = {
        "target_nodes": len(test_node_indices),
        "budget": budget,
        "attack_strategy": "under",
        "beam_width": 3,
        "base_model": "CompatibleGCN",
        "training_epochs": 200,
        "learning_rate": 0.01,
        "weight_decay": 5e-4
    }

    attack_results = run_attack_on_nodes(
        attack_model=calibrated_model,  # Use calibrated model for gradient computation
        calibrated_model=calibrated_model,  # Attack the calibrated model
        features=features,
        adj=adj,
        labels=labels,
        target_nodes=test_node_indices,
        budget=budget
    )

    # Analyze attack results
    print("\n" + "="*50)
    print("ATTACK ANALYSIS")
    print("="*50)

    successful_attacks = [r for r in attack_results if r['label_preserved'] and r['conf_change'] < 0]
    avg_conf_reduction = np.mean([r['conf_change'] for r in successful_attacks]) if successful_attacks else 0
    avg_perturbations = np.mean([r['perturbations'] for r in attack_results])

    print(f"Attack success rate: {len(successful_attacks)}/{len(attack_results)} ({len(successful_attacks)/len(attack_results)*100:.1f}%)")
    print(f"Average confidence reduction: {avg_conf_reduction:.4f}")
    print(f"Average perturbations used: {avg_perturbations:.1f}")

    # Show some example results
    print("\nExample attack results:")
    for result in attack_results[:5]:
        status = "SUCCESS" if result['label_preserved'] and result['conf_change'] < 0 else "FAILED"
        print(f"Node {result['node']}: {result['original_conf']:.4f} → {result['attacked_conf']:.4f} "
              f"({result['conf_change']:+.4f}) [{status}]")

    # Save comprehensive logs
    print("\n" + "="*50)
    print("SAVING ATTACK LOGS")
    print("="*50)
    save_attack_log(args, attack_results, base_metrics, calibrated_metrics, experiment_config)


if __name__ == "__main__":
    main()