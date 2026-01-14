#!/usr/bin/env python3
"""
Benchmark: Comprehensive Calibration Methods with CompatibleGCN

This benchmark compares multiple calibration methods:
- Temperature Scaling (TS)
- Vector Scaling (VS)
- Matrix Scaling (MS)
- ETS (Ensemble Temperature Scaling)
- CaGCN (Calibration-aware Graph Convolutional Network)
- DCGC (Decisive Edge Learning for Graph Calibration)
- SimCalib
- WATS
- GATS
- GETS

The process involves:
1. Loading the Cora dataset
2. Training a CompatibleGCN model
3. Applying all calibration methods
4. Evaluating and comparing calibration performance
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import time

from src.gnn.model import CompatibleGCN
from calibration.TS import TemperatureScaling
from calibration.VS import VectorScaling
from calibration.MS import MatrixScaling
from calibration.SimCalib import SimCalib
from calibration.WATS import WATS
from calibration.GATS import GATSCalibrator
from calibration.GETS import GETSCalibrator
from calibration.ETS import ETS
from calibration.DCGC import DCGC
from calibration.CaGCN import CaGCN
from calibration.utils import accuracy, edge_index_to_torch_matrix
from utils.ece import calculate_average_ece


def load_cora_data():
    """Load and prepare Cora dataset."""
    dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]

    # Convert edge_index to adjacency matrix for CompatibleGCN
    num_nodes = data.x.size(0)
    adj_matrix = edge_index_to_torch_matrix(data.edge_index, num_nodes)

    return data.x, data.y, adj_matrix, data.train_mask, data.val_mask, data.test_mask, data.edge_index


def train_base_model(model, x, y, adj, train_mask, val_mask, epochs=200):
    """Train the base CompatibleGCN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x, y, adj = x.to(device), y.to(device), adj.to(device)
    train_mask, val_mask = train_mask.to(device), val_mask.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, adj)
        loss = F.nll_loss(F.log_softmax(out[train_mask], dim=1), y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(x, adj)
                val_pred = F.log_softmax(val_out[val_mask], dim=1)
                val_acc = accuracy(val_pred, y[val_mask])
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')
            model.train()

    return model


def evaluate_calibration(model, x, y, adj, test_mask, model_name="Model"):
    """Evaluate model predictions and calibration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move all tensors to the same device
    x = x.to(device)
    y = y.to(device)
    adj = adj.to(device)
    test_mask = test_mask.to(device)
    model = model.to(device)

    model.eval()

    with torch.no_grad():
        logits = model(x, adj)
        if hasattr(logits, 'exp'):  # log_softmax output
            probs = logits.exp()
        else:  # raw logits
            probs = F.softmax(logits, dim=1)

        test_probs = probs[test_mask]
        test_labels = y[test_mask]

        # Calculate accuracy
        pred_labels = torch.argmax(test_probs, dim=1)
        acc = (pred_labels == test_labels).float().mean()

        # Calculate confidence (max probability)
        confidence = torch.max(test_probs, dim=1)[0]
        avg_confidence = confidence.mean()

        # Calculate ECE
        n_classes = test_probs.shape[1]
        ece = calculate_average_ece(test_probs.cpu().numpy(), test_labels.cpu().numpy(), n_classes, logits=False)

        print(f"{model_name} Results:")
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Average Confidence: {avg_confidence:.4f}")
        print(f"  Expected Calibration Error (ECE): {ece:.4f}")

        return acc, avg_confidence, ece


def evaluate_calibration_probs(probs, y, test_mask, model_name="Model"):
    """Evaluate calibration given probability tensor."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    probs = probs.to(device)
    y = y.to(device)
    test_mask = test_mask.to(device)

    test_probs = probs[test_mask]
    test_labels = y[test_mask]

    # Calculate accuracy
    pred_labels = torch.argmax(test_probs, dim=1)
    acc = (pred_labels == test_labels).float().mean()

    # Calculate confidence (max probability)
    confidence = torch.max(test_probs, dim=1)[0]
    avg_confidence = confidence.mean()

    # Calculate ECE
    n_classes = test_probs.shape[1]
    ece = calculate_average_ece(test_probs.cpu().numpy(), test_labels.cpu().numpy(), n_classes, logits=False)

    print(f"{model_name} Results:")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Average Confidence: {avg_confidence:.4f}")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")

    return acc, avg_confidence, ece


def main():
    """Main benchmark function."""
    print("=== Comprehensive Calibration Methods Benchmark ===\n")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load data
    print("1. Loading Cora dataset...")
    x, y, adj, train_mask, val_mask, test_mask, edge_index = load_cora_data()
    n_classes = y.max().item() + 1
    print(f"   Nodes: {x.size(0)}, Features: {x.size(1)}, Classes: {n_classes}")

    # Create and train base model
    print("\n2. Training CompatibleGCN model...")
    nfeat = x.size(1)
    model = CompatibleGCN(nfeat=nfeat, dataset_name='cora', nhid=64, dropout=0.5)
    trained_model = train_base_model(model, x, y, adj, train_mask, val_mask)

    # Get base model logits for calibration methods
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y, adj = x.to(device), y.to(device), adj.to(device)
    edge_index = edge_index.to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)

    trained_model.eval()
    with torch.no_grad():
        base_logits = trained_model(x, adj)

    # Evaluate base model
    print("\n3. Evaluating base model...")
    base_acc, base_conf, base_ece = evaluate_calibration(trained_model, x, y, adj, test_mask, "Base Model")

    results = {}
    results['Base Model'] = {'acc': base_acc, 'conf': base_conf, 'ece': base_ece, 'time': 0.0, 'params': 0}

    # Temperature Scaling
    print("\n4. Applying Temperature Scaling...")
    start_time = time.time()
    ts_model = TemperatureScaling(
        base_model=trained_model,
        x=x,
        y=y,
        adj=adj,
        val_idx=val_mask
    )
    ts_time = time.time() - start_time
    ts_acc, ts_conf, ts_ece = evaluate_calibration(ts_model, x, y, adj, test_mask, "Temperature Scaling")
    results['Temperature Scaling'] = {'acc': ts_acc, 'conf': ts_conf, 'ece': ts_ece, 'time': ts_time, 'params': 1}

    # Vector Scaling
    print("\n5. Applying Vector Scaling...")
    start_time = time.time()
    vs_model = VectorScaling(
        base_model=trained_model,
        x=x,
        y=y,
        adj=adj,
        val_idx=val_mask
    )
    vs_time = time.time() - start_time
    vs_acc, vs_conf, vs_ece = evaluate_calibration(vs_model, x, y, adj, test_mask, "Vector Scaling")
    results['Vector Scaling'] = {'acc': vs_acc, 'conf': vs_conf, 'ece': vs_ece, 'time': vs_time, 'params': n_classes}

    # Matrix Scaling
    print("\n6. Applying Matrix Scaling...")
    start_time = time.time()
    ms_model = MatrixScaling(
        base_model=trained_model,
        x=x,
        y=y,
        adj=adj,
        val_idx=val_mask
    )
    ms_time = time.time() - start_time
    ms_acc, ms_conf, ms_ece = evaluate_calibration(ms_model, x, y, adj, test_mask, "Matrix Scaling")
    results['Matrix Scaling'] = {'acc': ms_acc, 'conf': ms_conf, 'ece': ms_ece, 'time': ms_time, 'params': n_classes*n_classes + n_classes}

    # ETS (Ensemble Temperature Scaling)
    print("\n7. Applying ETS...")
    start_time = time.time()
    try:
        ets_model = ETS(
            trained_model,
            x,
            y,
            adj,
            val_mask,
        )
        ets_model.fit([train_mask, val_mask, test_mask])
        ets_time = time.time() - start_time
        ets_acc, ets_conf, ets_ece = evaluate_calibration(ets_model, x, y, adj, test_mask, "ETS")
        results['ETS'] = {'acc': ets_acc, 'conf': ets_conf, 'ece': ets_ece, 'time': ets_time, 'params': 3}
    except Exception as e:
        print(f"  ETS failed: {e}")
        results['ETS'] = {'acc': 0.0, 'conf': 0.0, 'ece': 1.0, 'time': 0.0, 'params': 'Failed'}

    # CaGCN
    print("\n8. Applying CaGCN...")
    start_time = time.time()
    try:
        cagcn_model = CaGCN(
            trained_model,
            x,
            y,
            adj,
            val_mask
        )
        cagcn_time = time.time() - start_time
        cagcn_acc, cagcn_conf, cagcn_ece = evaluate_calibration(cagcn_model, x, y, adj, test_mask, "CaGCN")
        results['CaGCN'] = {'acc': cagcn_acc, 'conf': cagcn_conf, 'ece': cagcn_ece, 'time': cagcn_time, 'params': 'Variable'}
    except Exception as e:
        print(f"  CaGCN failed: {e}")
        results['CaGCN'] = {'acc': 0.0, 'conf': 0.0, 'ece': 1.0, 'time': 0.0, 'params': 'Failed'}

    # DCGC
    print("\n9. Applying DCGC...")
    start_time = time.time()
    try:
        dcgc_model = DCGC(
            base_model=trained_model,
            features=x,
            labels=y,
            adj=adj,
            val_mask=val_mask
        )
        dcgc_time = time.time() - start_time
        dcgc_acc, dcgc_conf, dcgc_ece = evaluate_calibration(dcgc_model, x, y, adj, test_mask, "DCGC")
        results['DCGC'] = {'acc': dcgc_acc, 'conf': dcgc_conf, 'ece': dcgc_ece, 'time': dcgc_time, 'params': 'Variable'}
    except Exception as e:
        print(f"  DCGC failed: {e}")
        results['DCGC'] = {'acc': 0.0, 'conf': 0.0, 'ece': 1.0, 'time': 0.0, 'params': 'Failed'}

    # SimCalib
    print("\n10. Applying SimCalib...")
    start_time = time.time()
    try:
        simcalib_model = SimCalib(
            base_model=trained_model,
            features=x,
            labels=y,
            adj=adj,
            val_mask=val_mask,
            k=10,
            epsilon=1e-8
        )
        simcalib_time = time.time() - start_time
        simcalib_acc, simcalib_conf, simcalib_ece = evaluate_calibration(
            simcalib_model, x, y, adj, test_mask, "SimCalib"
        )
        results['SimCalib'] = {'acc': simcalib_acc, 'conf': simcalib_conf, 'ece': simcalib_ece, 'time': simcalib_time, 'params': 0}
    except Exception as e:
        print(f"  SimCalib failed: {e}")
        results['SimCalib'] = {'acc': 0.0, 'conf': 0.0, 'ece': 1.0, 'time': 0.0, 'params': 'Failed'}

    # WATS
    print("\n11. Applying WATS...")
    start_time = time.time()
    try:
        wats_model = WATS(
            base_model=trained_model,
            features=x,
            labels=y,
            adj=adj,
            val_mask=val_mask
        )
        wats_time = time.time() - start_time
        wats_acc, wats_conf, wats_ece = evaluate_calibration(wats_model, x, y, adj, test_mask, "WATS")
        results['WATS'] = {'acc': wats_acc, 'conf': wats_conf, 'ece': wats_ece, 'time': wats_time, 'params': 'Variable'}
    except Exception as e:
        print(f"  WATS failed: {e}")
        results['WATS'] = {'acc': 0.0, 'conf': 0.0, 'ece': 1.0, 'time': 0.0, 'params': 'Failed'}

    # GATS
    print("\n12. Applying GATS...")
    start_time = time.time()
    try:
        gats_model = GATSCalibrator(
            trained_model,
            x,
            y,
            adj,
            val_mask,
            heads=4,  # Reduced for faster benchmarking
            bias=1.0,
            bfs_depth=2
        )
        gats_time = time.time() - start_time
        gats_acc, gats_conf, gats_ece = evaluate_calibration(gats_model, x, y, adj, test_mask, "GATS")
        results['GATS'] = {'acc': gats_acc, 'conf': gats_conf, 'ece': gats_ece, 'time': gats_time, 'params': 'Variable'}
    except Exception as e:
        print(f"  GATS failed: {e}")
        results['GATS'] = {'acc': 0.0, 'conf': 0.0, 'ece': 1.0, 'time': 0.0, 'params': 'Failed'}

    # GETS
    print("\n13. Applying GETS...")
    start_time = time.time()
    try:
        conf = {
            'lr': 0.01,
            'max_epoch': 100,
            'patience': 10,
            'weight_decay': 5e-4
        }
        gets_model = GETSCalibrator(
            base_model=trained_model,
            num_classes=n_classes,
            device=device,
            conf=conf,
            x=x,
            y=y,
            adj=adj,
            val_idx=val_mask,
            num_experts=3,
            expert_select=2,
            hidden_dim=32,  # Reduced for faster benchmarking
            dropout_rate=0.1,
            num_layers=2,
            feature_hidden_dim=16,
            degree_hidden_dim=8,
            noisy_gating=True,
            loss_coef=1e-2,
            backbone='gcn'  # Using GCN backbone for consistency
        )
        gets_model.fit(adj, x, y, [train_mask, val_mask, test_mask])
        gets_time = time.time() - start_time
        gets_acc, gets_conf, gets_ece = evaluate_calibration(gets_model, x, y, adj, test_mask, "GETS")
        results['GETS'] = {'acc': gets_acc, 'conf': gets_conf, 'ece': gets_ece, 'time': gets_time, 'params': 'Variable'}
    except Exception as e:
        print(f"  GETS failed: {e}")
        results['GETS'] = {'acc': 0.0, 'conf': 0.0, 'ece': 1.0, 'time': 0.0, 'params': 'Failed'}

    # Comprehensive Results Summary
    print(f"\n=== Comprehensive Results Summary ===")
    print(f"{'Method':<18} {'Accuracy':<10} {'Confidence':<12} {'ECE':<8} {'ECE Reduction':<15} {'Time (s)':<10} {'Complexity':<12}")
    print(f"{'-'*110}")

    for method_name, result in results.items():
        if method_name == 'Base Model':
            improvement = '-'
        else:
            if result['ece'] < 1.0:  # Valid result
                improvement = f"{(base_ece - result['ece']) / base_ece * 100:+.1f}%"
            else:
                improvement = 'Failed'

        print(f"{method_name:<18} {result['acc']:.4f}     {result['conf']:.4f}       {result['ece']:.4f}   {improvement:<15} {result['time']:<10.2f} {str(result['params']):<12}")

    # Best method analysis
    print(f"\n=== Method Comparison ===")
    valid_methods = [(name, result['ece']) for name, result in results.items()
                    if name != 'Base Model' and result['ece'] < 1.0]

    if valid_methods:
        best_method = min(valid_methods, key=lambda x: x[1])
        print(f"🏆 {best_method[0]} achieved the best calibration (ECE: {best_method[1]:.4f})")

        # Efficiency analysis
        print(f"\n=== Efficiency Analysis ===")
        for method_name, result in results.items():
            if method_name != 'Base Model' and result['ece'] < 1.0:
                improvement = (base_ece - result['ece']) / base_ece * 100
                efficiency = improvement / result['time'] if result['time'] > 0 else 0
                print(f"{method_name}: {improvement:+.1f}% improvement in {result['time']:.2f}s = {efficiency:.2f} improvement/second")
    else:
        print("⚠️  All calibration methods failed or performed worse than base model")


if __name__ == "__main__":
    main()