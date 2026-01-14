import numpy as np
import scipy.sparse as sp
import torch

def to_scipy(matrix):
    """Convert a torch tensor or numpy array to a SciPy CSR matrix."""
    if isinstance(matrix, sp.spmatrix):
        return matrix
    if isinstance(matrix, torch.Tensor):
        if matrix.is_sparse:
            matrix = matrix.coalesce().cpu()
            values = matrix.values().numpy()
            indices = matrix.indices().numpy()
            return sp.csr_matrix((values, indices), shape=matrix.shape)
        return sp.csr_matrix(matrix.detach().cpu().numpy())
    if isinstance(matrix, np.ndarray):
        return sp.csr_matrix(matrix)
    raise TypeError(f"Unsupported type for to_scipy conversion: {type(matrix)}")
