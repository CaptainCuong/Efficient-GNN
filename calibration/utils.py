import torch
import numpy as np
import scipy.sparse as sp

def edge_index_to_torch_matrix(edge_index, num_nodes):
    """
    Converts a graph represented by edge indices into an adjacency matrix using PyTorch.

    Parameters:
    - edge_index (torch.Tensor): A 2D tensor of shape (2, num_edges) containing the edge indices.
        The first row contains the source nodes, and the second row contains the target nodes.
    - num_nodes (int): The total number of nodes in the graph.

    Returns:
    - torch.Tensor: A 2D tensor of shape (num_nodes, num_nodes) representing the adjacency matrix.
        The value at position (i, j) is 1 if there is an edge from node i to node j, and 0 otherwise.
    """
    # Initialize an N x N tensor with zeros
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # Populate the tensor using the edge_index
    src_nodes = edge_index[0]
    tgt_nodes = edge_index[1]

    adj_matrix[src_nodes, tgt_nodes] = 1  # Set 1 for each edge

    return adj_matrix

def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    # if hasattr(tensor, 'nnz'):
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False

def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    # TODO: maybe using coo format would be better?
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0 :
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
        adj : torch.tensor
            Square adjacency matrix
    """
    device = adj.device
    if sparse:
        # warnings.warn('If you find the training process is too slow, you can uncomment line 207 in deeprobust/graph/utils.py. Note that you need to install torch_sparse')
        # TODO if this is too slow, uncomment the following code,
        # but you need to install torch_scatter
        # return normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx).to(device)
    else:
        mx = adj + torch.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx

# TO DO: convert a square adjacency matrix to a sparse tensor
def to_sparse_tensor(adj):
    """Convert a square adjacency matrix to a sparse tensor.

    Parameters
    ----------
    adj : torch.tensor
        Square adjacency matrix

    Returns
    -------
    torch.sparse.FloatTensor
        Sparse adjacency matrix
    """
    adj_t = adj.to_sparse()
    print(adj_t)
    return adj_t[0]

def accuracy(outputs, labels):
    """
    Evaluate the accuracy of a GNN's predictions.

    Parameters:
    outputs (torch.Tensor): Outputs returned by the GNN of shape [num_nodes, num_classes].
    labels (torch.Tensor): Ground truth labels of shape [num_nodes].

    Returns:
    float: Accuracy of the predictions as a value between 0 and 1.
    """
    # Check if outputs and labels have numpy type
    if not isinstance(outputs, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("Input arrays must be of type torch.Tensor.")
    
    # Check if outputs and labels have the same number of elements
    if outputs.shape[0] != labels.shape[0]:
        raise ValueError("Input arrays must have the same number of elements.")
    
    # Get the predicted labels by taking the argmax of outputs along the class dimension
    predicted_labels = torch.argmax(outputs, dim=1)

    # Compare predicted labels with true labels
    correct_predictions = torch.sum(predicted_labels == labels)

    # Calculate accuracy
    accuracy = correct_predictions / labels.shape[0]

    return accuracy.item()

def shortest_path_length(edge_index, mask, max_hop):
    """
    Return the shortest path length to the mask for every node
    """
    # Set device to match edge_index
    edge_index, mask = edge_index.to(edge_index.device), mask.to(edge_index.device)
    dist_to_train = torch.ones_like(mask, dtype=torch.long, device=mask.device) * torch.iinfo(torch.long).max
    seen_mask = torch.clone(mask)
    for hop in range(max_hop):
        current_hop = torch.nonzero(mask)
        dist_to_train[mask] = hop
        next_hop = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)
        for node in current_hop:
            print(node)
            raise
            node_mask = edge_index[0,:]==node
            print(edge_index)
            print(node)
            print(edge_index.shape)
            raise
            print(node_mask)
            nbrs = edge_index[1,node_mask]
            next_hop[nbrs] = True
        hop += 1
        # mask for the next hop shouldn't be seen before
        mask = torch.logical_and(next_hop, ~seen_mask)
        seen_mask[next_hop] = True
    return dist_to_train