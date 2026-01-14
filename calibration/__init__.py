"""
Calibration methods for improving model confidence estimation.

This module provides implementations of various calibration techniques
for neural networks, particularly focused on Graph Neural Networks.

Available calibration methods:
- Temperature Scaling (TS): Simple and effective single-parameter scaling
- Vector Scaling (VS): Multi-parameter scaling with class-specific temperatures
- Matrix Scaling (MS): Full affine transformation of logits
- Ensemble Temperature Scaling (ETS): Enhanced temperature scaling with ensemble weights
- CaGCN: Calibration-aware Graph Convolutional Network
- WATS: Wavelet-based Adaptive Temperature Scaling
- GATS: Graph Attention-based Temperature Scaling
- GETS: Graph Expert Temperature Scaling with mixture of experts
- SimCalib: Similarity-based calibration for graph neural networks
"""

# Temperature Scaling
from .TS import (
    TemperatureScaling,
    calibrate_with_temperature_scaling,
    evaluate_calibration,
    plot_reliability_diagram,
    plot_confidence_histogram,
    comprehensive_calibration_analysis,
    load_calibrated_model
)

# Vector Scaling
from .VS import (
    VectorScaling,
    calibrate_with_vector_scaling
)

# Matrix Scaling
from .MS import (
    MatrixScaling,
    calibrate_with_matrix_scaling
)

# Ensemble Temperature Scaling
from .ETS import ETS

# Calibration-aware GCN
from .CaGCN import CaGCN

# Wavelet-based Adaptive Temperature Scaling
from .WATS import (
    WATS
)

# Graph Attention-based Temperature Scaling
from .GATS import (
    GATSCalibrator,
    CalibAttentionLayer,
    calibrate_with_gats
)

# Graph Expert Temperature Scaling
from .GETS import (
    GETSCalibrator,
    GCN_Expert,
    GAT_Expert,
    GIN_Expert,
    calibrate_with_gets
)

# Similarity-based Calibration
from .SimCalib import (
    SimCalib
)

__all__ = [
    # Temperature Scaling
    "TemperatureScaling",
    "calibrate_with_temperature_scaling",
    "evaluate_calibration",
    "plot_reliability_diagram",
    "plot_confidence_histogram",
    "comprehensive_calibration_analysis",
    "load_calibrated_model",

    # Vector Scaling
    "VectorScaling",
    "calibrate_with_vector_scaling",

    # Matrix Scaling
    "MatrixScaling",
    "calibrate_with_matrix_scaling",

    # Ensemble Temperature Scaling
    "ETS",

    # Calibration-aware GCN
    "CaGCN",

    # WATS
    "WATS",

    # GATS
    "GATSCalibrator",
    "CalibAttentionLayer",
    "calibrate_with_gats",

    # GETS
    "GETSCalibrator",
    "GCN_Expert",
    "GAT_Expert",
    "GIN_Expert",
    "calibrate_with_gets",

    # SimCalib
    "SimCalib",
    "SimCalibConfig",
    "MahalanobisPrototypes",
    "GFeat",
    "MovementSimilarity",
    "expected_calibration_error"
]