#!/usr/bin/env python3
"""
UGCA (Underconfidence Gradient Calibration Attack) with Basic Under Strategy for Multiple Datasets

This script demonstrates the UGCA attack using the basic attack function with "under" strategy
on calibrated GNNs across multiple datasets:
1. Training a base GNN model
2. Calibrating it with various calibration methods (CaGCN, ETS, GETS, MS, TS, VS, WATS, SimCalib)
3. Attacking the calibrated model using attack with strategy="under" from Calib_FGA

Supported datasets: Cora, CiteSeer, PubMed, CoraML, Ogbn-arxiv, Photo, Physics, Reddit
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
import time
from datetime import datetime
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CitationFull, Reddit

# Fix for PyTorch 2.6+ weights_only default change
# Add safe globals for torch_geometric data classes used by OGB
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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


# Dataset configurations
DATASET_INFO = {
    'cora': {'type': 'planetoid', 'name': 'Cora', 'num_classes': 7},
    'citeseer': {'type': 'planetoid', 'name': 'CiteSeer', 'num_classes': 6},
    'pubmed': {'type': 'planetoid', 'name': 'PubMed', 'num_classes': 3},
    'coraml': {'type': 'citationfull', 'name': 'Cora_ML', 'num_classes': 7},
    'ogbn-arxiv': {'type': 'ogb', 'name': 'ogbn-arxiv', 'num_classes': 40},
    'photo': {'type': 'amazon', 'name': 'Photo', 'num_classes': 8},
    'physics': {'type': 'coauthor', 'name': 'Physics', 'num_classes': 5},
    'reddit': {'type': 'reddit', 'name': 'Reddit', 'num_classes': 41},
}


def load_dataset(dataset_name, max_nodes=None):
    """
    Load specified dataset and return tensors for training and attacks.

    Args:
        dataset_name: Name of the dataset to load
        max_nodes: Maximum number of nodes to use (for large datasets like Reddit/Ogbn-arxiv)
    """
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_INFO.keys())}")

    info = DATASET_INFO[dataset_name_lower]
    dataset_type = info['type']

    print(f"Loading {dataset_name} dataset...")

    # Load dataset based on type
    if dataset_type == 'planetoid':
        dataset = Planetoid(root="/helios-storage/helios3-data/cuong/data", name=info['name'])
        data = dataset[0]
    elif dataset_type == 'amazon':
        dataset = Amazon(root="/helios-storage/helios3-data/cuong/data", name=info['name'])
        data = dataset[0]
        # Amazon datasets don't have predefined splits, create random splits
        data = create_random_splits(data)
    elif dataset_type == 'coauthor':
        dataset = Coauthor(root="/helios-storage/helios3-data/cuong/data", name=info['name'])
        data = dataset[0]
        # Coauthor datasets don't have predefined splits, create random splits
        data = create_random_splits(data)
    elif dataset_type == 'citationfull':
        dataset = CitationFull(root="/helios-storage/helios3-data/cuong/data", name=info['name'])
        data = dataset[0]
        # CitationFull datasets don't have predefined splits, create random splits
        data = create_random_splits(data)
    elif dataset_type == 'ogb':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=info['name'], root="/helios-storage/helios3-data/cuong/data")
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        # Convert OGB splits to masks
        num_nodes = data.num_nodes
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True
        # Flatten labels if needed
        if data.y.dim() > 1:
            data.y = data.y.view(-1)
    elif dataset_type == 'reddit':
        dataset = Reddit(root="/helios-storage/helios3-data/cuong/data/Reddit")
        data = dataset[0]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # For large datasets, optionally subsample
    if max_nodes is not None and data.num_nodes > max_nodes:
        print(f"Subsampling dataset from {data.num_nodes} to {max_nodes} nodes...")
        data = subsample_graph(data, max_nodes)

    num_nodes = data.num_nodes
    print(f"Dataset has {num_nodes} nodes")

    # Build adjacency matrix (dense) - only feasible for smaller graphs
    if num_nodes > 50000:
        print(f"Warning: Dataset has {num_nodes} nodes. Using sparse adjacency representation.")
        # For large graphs, we'll work with edge_index directly
        adj = None
        edge_index = data.edge_index
    else:
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        edge_index = data.edge_index
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = adj + adj.t()  # Make symmetric
        adj = torch.clamp(adj, 0, 1)  # Ensure values are between 0 and 1
        adj.fill_diagonal_(1.0)  # Add self-loops

    features = data.x.float()
    labels = data.y

    num_features = dataset.num_features if hasattr(dataset, 'num_features') else features.shape[1]
    num_classes = info['num_classes']

    return data, features, adj, labels, num_features, num_classes, edge_index


def create_random_splits(data, train_ratio=0.6, val_ratio=0.2):
    """Create random train/val/test splits for datasets without predefined splits."""
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[indices[:train_size]] = True
    data.val_mask[indices[train_size:train_size + val_size]] = True
    data.test_mask[indices[train_size + val_size:]] = True

    return data


def subsample_graph(data, max_nodes):
    """Subsample a graph to max_nodes nodes while preserving structure."""
    num_nodes = data.num_nodes

    # Random node selection
    perm = torch.randperm(num_nodes)[:max_nodes]
    perm, _ = torch.sort(perm)  # Keep sorted for easier indexing

    # Create node mapping
    node_map = torch.full((num_nodes,), -1, dtype=torch.long)
    node_map[perm] = torch.arange(max_nodes)

    # Filter edges
    edge_mask = (node_map[data.edge_index[0]] >= 0) & (node_map[data.edge_index[1]] >= 0)
    new_edge_index = node_map[data.edge_index[:, edge_mask]]

    # Create new data object
    new_data = data.clone()
    new_data.x = data.x[perm]
    new_data.y = data.y[perm]
    new_data.edge_index = new_edge_index
    new_data.num_nodes = max_nodes

    # Update masks
    if hasattr(data, 'train_mask'):
        new_data.train_mask = data.train_mask[perm]
    if hasattr(data, 'val_mask'):
        new_data.val_mask = data.val_mask[perm]
    if hasattr(data, 'test_mask'):
        new_data.test_mask = data.test_mask[perm]

    return new_data


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


def save_attack_log(args, attack_results, base_metrics, calibrated_metrics, attacked_metrics, experiment_config, log_dir="./logs"):
    """Save comprehensive attack results and experiment information to log files."""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare comprehensive log data
    log_data = {
        "experiment_info": {
            "timestamp": timestamp,
            "attack_type": "UGCA (attack with under strategy)",
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
            },
            "after_attack": {
                "accuracy": attacked_metrics["accuracy"],
                "ece": attacked_metrics["ece"],
                "avg_confidence": attacked_metrics["avg_confidence"]
            }
        },
        "attack_summary": {
            "total_nodes_attacked": len(attack_results),
            "successful_attacks": len([r for r in attack_results if r['label_preserved'] and r['conf_change'] < 0]),
            "success_rate": len([r for r in attack_results if r['label_preserved'] and r['conf_change'] < 0]) / len(attack_results) if attack_results else 0,
            "avg_confidence_reduction": float(np.mean([r['conf_change'] for r in attack_results if r['label_preserved'] and r['conf_change'] < 0])) if any(r['label_preserved'] and r['conf_change'] < 0 for r in attack_results) else 0,
            "avg_perturbations": float(np.mean([r['perturbations'] for r in attack_results])) if attack_results else 0,
            "timing": {
                "total_time": float(np.sum([r['iteration_time'] for r in attack_results])) if attack_results else 0,
                "avg_attack_time": float(np.mean([r['attack_time'] for r in attack_results])) if attack_results else 0,
                "avg_iteration_time": float(np.mean([r['iteration_time'] for r in attack_results])) if attack_results else 0,
                "min_iteration_time": float(np.min([r['iteration_time'] for r in attack_results])) if attack_results else 0,
                "max_iteration_time": float(np.max([r['iteration_time'] for r in attack_results])) if attack_results else 0
            }
        },
        "detailed_results": attack_results
    }

    # Save JSON log
    json_file = os.path.join(log_dir, f"ugca_under_{args.dataset.lower()}_{args.calibration_method.lower()}_attack_log_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    # Save human-readable summary
    summary_file = os.path.join(log_dir, f"ugca_under_{args.dataset.lower()}_{args.calibration_method.lower()}_attack_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("UGCA UNDER ATTACK EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Attack Type: UGCA (attack with under strategy)\n")
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

        f.write(f"After Attack:\n")
        f.write(f"  Accuracy: {attacked_metrics['accuracy']:.4f}\n")
        f.write(f"  ECE: {attacked_metrics['ece']:.4f}\n")
        f.write(f"  Avg Confidence: {attacked_metrics['avg_confidence']:.4f}\n")
        f.write(f"  ECE Change: {attacked_metrics['ece'] - calibrated_metrics['ece']:+.4f}\n")
        f.write("\n")

        f.write("ATTACK RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total nodes attacked: {len(attack_results)}\n")
        f.write(f"Successful attacks: {log_data['attack_summary']['successful_attacks']}\n")
        f.write(f"Success rate: {log_data['attack_summary']['success_rate']:.2%}\n")
        f.write(f"Avg confidence reduction: {log_data['attack_summary']['avg_confidence_reduction']:.4f}\n")
        f.write(f"Avg perturbations: {log_data['attack_summary']['avg_perturbations']:.1f}\n\n")

        f.write("TIMING STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total time: {log_data['attack_summary']['timing']['total_time']:.2f}s\n")
        f.write(f"Average attack time per node: {log_data['attack_summary']['timing']['avg_attack_time']:.4f}s\n")
        f.write(f"Average iteration time per node: {log_data['attack_summary']['timing']['avg_iteration_time']:.4f}s\n")
        f.write(f"Min iteration time: {log_data['attack_summary']['timing']['min_iteration_time']:.4f}s\n")
        f.write(f"Max iteration time: {log_data['attack_summary']['timing']['max_iteration_time']:.4f}s\n\n")

        f.write("SAMPLE RESULTS:\n")
        f.write("-" * 40 + "\n")
        for result in attack_results[:10]:
            status = "SUCCESS" if result['label_preserved'] and result['conf_change'] < 0 else "FAILED"
            f.write(f"Node {result['node']}: {result['original_conf']:.4f} -> {result['attacked_conf']:.4f} "
                   f"({result['conf_change']:+.4f}) [{status}]\n")

    print(f"\nAttack logs saved:")
    print(f"  JSON log: {json_file}")
    print(f"  Summary: {summary_file}")

    return json_file, summary_file


def run_attack_on_nodes(attack_model, calibrated_model, features, adj, labels, target_nodes, budget=5):
    """Run attack with under strategy on specified nodes."""
    print(f"\nRunning UGCA (attack with under strategy) on {len(target_nodes)} nodes with budget {budget}...")

    attack_results = []
    successful_attacks = 0

    for i, node in enumerate(target_nodes):
        iteration_start_time = time.time()
        print(f"Attacking node {node} ({i+1}/{len(target_nodes)})...")

        # Reset to original adjacency for each node (no cumulative attacks)
        modified_adj = adj.clone()

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
        attack_start_time = time.time()
        attack = Calib_FGA(attack_model, device=features.device)

        n_perturb = []
        best_conf = []

        attack.attack(
            ori_features=features,
            ori_adj=modified_adj,
            target_node=node,
            n_perturbations=budget,
            strategy="under",
            res_gt=labels,
            verbose=False,
            n_perturb=n_perturb,
            best_conf=best_conf
        )
        attack_time = time.time() - attack_start_time

        # Update modified_adj with the attack result
        modified_adj = attack.modified_adj.clone()

        # Evaluate attacked model
        with torch.no_grad():
            attacked_logits = calibrated_model(features, modified_adj)
            if hasattr(attacked_logits, 'exp'):
                attacked_probs = attacked_logits.exp()
            else:
                attacked_probs = F.softmax(attacked_logits, dim=1)
            attacked_pred = attacked_probs[node].argmax().item()
            attacked_conf = attacked_probs[node].max().item()

        if original_pred != attacked_pred:
            raise ValueError(f"Attack failed: original ({original_pred}) and attacked ({attacked_pred}) predictions do not match.")

        iteration_time = time.time() - iteration_start_time

        # Store results
        result = {
            'node': int(node),
            'original_pred': int(original_pred),
            'original_conf': float(original_conf),
            'attacked_pred': int(attacked_pred),
            'attacked_conf': float(attacked_conf),
            'conf_change': float(attacked_conf - original_conf),
            'perturbations': int(n_perturb[0] if n_perturb else 0),
            'label_preserved': bool(original_pred == attacked_pred),
            'attack_time': float(attack_time),
            'iteration_time': float(iteration_time)
        }
        attack_results.append(result)

        if result['label_preserved'] and result['conf_change'] < 0:
            successful_attacks += 1

        print(f"  Original: pred={original_pred}, conf={original_conf:.4f}")
        print(f"  Attacked: pred={attacked_pred}, conf={attacked_conf:.4f}")
        print(f"  Change: {result['conf_change']:+.4f}, Perturbations: {result['perturbations']}")
        print(f"  Attack time: {attack_time:.4f}s, Total iteration time: {iteration_time:.4f}s")

        if original_pred != attacked_pred:
            print(original_probs)
            print(attacked_probs)
            raise ValueError("Attack failed: original and attacked predictions do not match.")
        
    # Calculate timing statistics
    attack_times = [r['attack_time'] for r in attack_results]
    iteration_times = [r['iteration_time'] for r in attack_results]
    total_time = sum(iteration_times)
    avg_attack_time = np.mean(attack_times)
    avg_iteration_time = np.mean(iteration_times)

    print(f"\nAttack Summary:")
    print(f"Successful attacks (label preserved + confidence reduced): {successful_attacks}/{len(target_nodes)}")
    print(f"\nTiming Statistics:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average attack time per node: {avg_attack_time:.4f}s")
    print(f"  Average iteration time per node: {avg_iteration_time:.4f}s")

    return attack_results, modified_adj


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='UGCA Under Attack for Multiple Datasets')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed', 'CoraML', 'Ogbn-arxiv', 'Photo', 'Physics', 'Reddit'],
                        help='Dataset to use (default: Cora)')
    parser.add_argument('--calibration-method', '--calib-method',
                        choices=['CaGCN', 'DCGC', 'ETS', 'GATS', 'GETS', 'MS', 'TS', 'VS', 'WATS', 'SimCalib'],
                        default='TS',
                        help='Calibration method to use (default: TS)')
    parser.add_argument('--budget', type=int, default=5,
                        help='Number of perturbations allowed for attack (default: 5)')

    # Attack configuration arguments
    parser.add_argument('--attack-nodes', type=int, default=100,
                        help='Number of nodes to attack (default: 100)')

    # Large dataset handling
    parser.add_argument('--max-nodes', type=int, default=None,
                        help='Maximum nodes to use for large datasets (default: None, use all)')

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
    print(f"Using dataset: {args.dataset}")
    print(f"Using calibration method: {args.calibration_method}")

    # Set default max_nodes for large datasets
    if args.max_nodes is None:
        if args.dataset.lower() in ['pubmed', 'ogbn-arxiv', 'photo', 'physics', 'reddit']:
            args.max_nodes = 5000  # Default limit for large datasets
            print(f"Large dataset detected. Limiting to {args.max_nodes} nodes. Use --max-nodes to change.")

    # Load data
    data, features, adj, labels, num_features, num_classes, edge_index = load_dataset(
        args.dataset, max_nodes=args.max_nodes
    )
    print(f"Loaded {args.dataset}: {data.num_nodes} nodes, {num_classes} classes, {num_features} features")

    # Check if dense adjacency is available
    if adj is None:
        raise ValueError(f"Dataset {args.dataset} is too large for dense adjacency matrix. "
                        f"Use --max-nodes to limit the number of nodes.")

    # Move to device
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)

    # Step 1: Train base model
    # Pass nclass explicitly since some dataset names may not be in CompatibleGCN's mapping
    base_model = CompatibleGCN(num_features, nclass=num_classes).to(device)
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
    print("ATTACKING CALIBRATED MODEL WITH UGCA (UNDER STRATEGY)")
    print("="*50)

    # Select test nodes for attack
    available_test_nodes = torch.nonzero(data.test_mask).flatten()
    num_attack_nodes = min(args.attack_nodes, len(available_test_nodes))
    test_node_indices = available_test_nodes[:num_attack_nodes].cpu().numpy()
    budget = args.budget

    print(f"Selected {num_attack_nodes} nodes for attack (out of {len(available_test_nodes)} test nodes)")

    # Experiment configuration for logging
    experiment_config = {
        "dataset": args.dataset,
        "num_nodes": data.num_nodes,
        "num_features": num_features,
        "num_classes": num_classes,
        "target_nodes": len(test_node_indices),
        "budget": budget,
        "attack_strategy": "under",
        "attack_method": "attack",
        "loss_function": "underconfidence_objective",
        "base_model": "CompatibleGCN",
        "training_epochs": 200,
        "learning_rate": 0.01,
        "weight_decay": 5e-4
    }

    attack_results, modified_adj = run_attack_on_nodes(
        attack_model=calibrated_model,  # Use calibrated model for gradient computation
        calibrated_model=calibrated_model,  # Attack the calibrated model
        features=features,
        adj=adj,
        labels=labels,
        target_nodes=test_node_indices,
        budget=budget
    )

    # Evaluate model after attack
    print("\n" + "="*50)
    print("EVALUATING MODEL AFTER ATTACK")
    print("="*50)
    attacked_acc, attacked_ece, attacked_conf = evaluate_model_calibration(
        calibrated_model, features, modified_adj, labels, data.test_mask, "After Attack"
    )

    attacked_metrics = {
        "accuracy": attacked_acc,
        "ece": attacked_ece,
        "avg_confidence": attacked_conf,
    }

    print(f"\nECE change after attack: {attacked_ece - calib_ece:+.4f}")

    # Analyze attack results
    print("\n" + "="*50)
    print("ATTACK ANALYSIS")
    print("="*50)

    successful_attacks = [r for r in attack_results if r['label_preserved'] and r['conf_change'] < 0]
    avg_conf_reduction = np.mean([r['conf_change'] for r in successful_attacks]) if successful_attacks else 0
    avg_perturbations = np.mean([r['perturbations'] for r in attack_results])

    # Timing statistics
    total_time = np.sum([r['iteration_time'] for r in attack_results])
    avg_attack_time = np.mean([r['attack_time'] for r in attack_results])
    avg_iteration_time = np.mean([r['iteration_time'] for r in attack_results])

    print(f"Dataset: {args.dataset}")
    print(f"Attack success rate: {len(successful_attacks)}/{len(attack_results)} ({len(successful_attacks)/len(attack_results)*100:.1f}%)")
    print(f"Average confidence reduction: {avg_conf_reduction:.4f}")
    print(f"Average perturbations used: {avg_perturbations:.1f}")
    print(f"\nTiming Statistics:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average attack time per node: {avg_attack_time:.4f}s")
    print(f"  Average iteration time per node: {avg_iteration_time:.4f}s")

    # Show some example results
    print("\nExample attack results:")
    for result in attack_results[:5]:
        status = "SUCCESS" if result['label_preserved'] and result['conf_change'] < 0 else "FAILED"
        print(f"Node {result['node']}: {result['original_conf']:.4f} -> {result['attacked_conf']:.4f} "
              f"({result['conf_change']:+.4f}) [{status}]")

    # Save comprehensive logs
    print("\n" + "="*50)
    print("SAVING ATTACK LOGS")
    print("="*50)
    save_attack_log(args, attack_results, base_metrics, calibrated_metrics, attacked_metrics, experiment_config)


if __name__ == "__main__":
    main()
