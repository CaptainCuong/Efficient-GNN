# Graph Calibration Attack Methods

This repository implements various calibration attack methods for Graph Neural Networks (GNNs). Unlike traditional adversarial attacks that aim to flip predictions, these attacks manipulate model confidence while preserving predicted labels.

## Overview

Graph calibration attacks target the confidence scores of GNN predictions rather than the predictions themselves. The goal is to make a well-calibrated model either overconfident or underconfident in its predictions while maintaining prediction accuracy.

## Attack Methods

### 1. Calib_FGA (Fast Gradient Attack)
**File:** `calib_attack/calib_fga.py`

A gradient-based attack adapted from the Fast Gradient Attack (FGA) for calibration manipulation.

**Key Features:**
- **Basic Attack**: Iteratively selects edges based on gradient magnitude
- **Reranked Attack**: Enhanced with label-flip prevention using probability derivatives
- **Hybrid Loss Attack**: Adaptive loss switching between calibration and classification objectives
- **Beam Search Attack**: Most advanced variant using beam search with hybrid loss and reranking

**Attack Strategies:**
- `over`: Overconfidence attack (increase confidence)
- `under`: Underconfidence attack (decrease confidence)
- `under_kl`: KL-divergence with uniform distribution
- `target`: Targeted attack based on ground truth alignment
- `max`: Maximize confidence for predicted label

**Usage:**
```python
from calib_attack.calib_fga import Calib_FGA

attack = Calib_FGA(model, device='cpu')
attack.attack(features, adj, target_node=0, n_perturbations=5, strategy='under')
```

### 2. Calib_IGA (Integrated Gradients Attack)
**File:** `calib_attack/calib_iga.py`

Based on Integrated Gradients, this attack computes importance scores for edges and features to guide perturbations.

**Key Features:**
- Uses integrated gradients to compute edge/feature importance
- Supports both structure and feature attacks
- Iteratively applies perturbations based on importance scores
- Maintains label preservation during attacks

**Usage:**
```python
from calib_attack.calib_iga import Calib_IGA

attack = Calib_IGA(model, attack_structure=True, device='cpu')
attack.attack(features, adj, target_node=0, n_perturbations=5,
              strategy='under_kl', steps=10)
```

### 3. Calib_Random (Random Attack)
**File:** `calib_attack/calib_random.py`

A baseline random attack that randomly selects edges/features to perturb.

**Key Features:**
- Random perturbation selection
- Only applies perturbations that improve calibration objective
- Serves as baseline for comparison with gradient-based methods
- Supports maximum trial limits per perturbation

**Usage:**
```python
from calib_attack.calib_random import Calib_Random

attack = Calib_Random(model, attack_structure=True, device='cpu')
attack.attack(features, adj, target_node=0, n_perturbations=5,
              strategy='under', max_trials=100)
```

### 4. Calib_RND (Random Node-Focused Attack)
**File:** `calib_attack/calib_rnd.py`

An enhanced random attack with additional node injection capabilities.

**Key Features:**
- Random edge/feature perturbations focused on target node
- Node injection attack capability
- Multiple random trials per perturbation step
- Preserves label while optimizing calibration objectives

**Usage:**
```python
from calib_attack.calib_rnd import Calib_RND

attack = Calib_RND(model, attack_structure=True, device='cpu')
# Standard attack
attack.attack(features, adj, target_node=0, n_perturbations=5, strategy='under_kl')

# Node injection attack
attack.random_node_injection(features, adj, target_node=0, n_injected=2,
                           n_perturbations=5, strategy='under_kl')
```

## Attack Strategies

All attack methods support multiple strategies:

- **`under`**: Underconfidence attack using standard confidence loss
- **`over`**: Overconfidence attack to increase confidence scores
- **`under_kl`**: Underconfidence using KL-divergence with uniform distribution
- **`target`**: Targeted attack based on ground truth label alignment

## Calibration Loss Functions

The attacks use various loss functions defined in `calib_attack/calib_attack_loss.py`:

- `underconfidence_objective`: Minimize confidence in predicted label
- `overconfidence_objective`: Maximize confidence in predicted label
- `kl_divergence_with_uniform`: KL divergence between predictions and uniform distribution
- `kl_divergence_target`: Targeted KL divergence loss

## Key Innovations

### 1. Label Preservation
All attacks maintain prediction accuracy while manipulating confidence scores.

### 2. Reranking Mechanism
Advanced attacks use probability derivatives to predict and prevent label flips.

### 3. Hybrid Loss Functions
Adaptive switching between calibration loss (when label unchanged) and classification loss (for label restoration).

### 4. Beam Search
Maintains multiple candidate perturbations to avoid local optima.

## Experimental Framework

### Reproduce Table 2 (Attack Method Comparison):
- `baseline_bench.py`: The naive attack
- `kl_loss_bench.py`: The naive attack with KL loss
- `rerank_bench.py`: The naive attack + KL loss + Reranking technique
- `rerank_hybridloss_bench.py`: The naive attack + KL loss + Reranking technique + Loss selection technique
- `unified_framework_bench.py`: The naive attack + KL loss + Reranking technique + Loss selection + Beam search

### Reproduce Table 3:
[Details to be added]

### Reproduce Figure 5 (Budget Analysis):
- `budget_bench.py`

### Reproduce Figure 6 (Generalization Analysis):
- `generalization_bench.py`

## Dependencies

- PyTorch
- PyTorch Geometric
- SciPy
- NumPy
- Tabulate (for result visualization)

## Citation

This implementation is based on research into adversarial attacks on graph neural networks, particularly focusing on calibration manipulation rather than traditional prediction-flipping attacks.

## Usage Notes

1. **Device Compatibility**: All attacks support both CPU and GPU execution
2. **Input Formats**: Support both dense and sparse matrix formats
3. **Progress Tracking**: Verbose mode available for detailed attack progress
4. **Result Tracking**: Built-in tracking for perturbation counts and confidence scores
5. **Memory Optimization**: Implements `torch.no_grad()` for memory-efficient evaluation

## Defensive Considerations

These attack implementations are designed for:
- Security research and analysis
- Understanding GNN vulnerabilities
- Developing defensive calibration methods
- Benchmarking model robustness

**Note**: This code is intended for defensive security research only. It should not be used for malicious purposes.
