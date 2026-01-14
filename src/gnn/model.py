"""Model definitions for scalable message passing."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

class CompatibleGCN(nn.Module):
    """Two-layer GCN compatible with Calib_FGA expectations."""

    # Hardcoded dataset class counts
    DATASET_CLASSES = {
        'cora': 7,
        'citeseer': 6,
        'pubmed': 3,
        'reddit': 41,
        'amazon-computers': 10,
        'amazon-photo': 8,
        'coauthor-cs': 15,
        'coauthor-physics': 5,
        'dblp': 4,
        'ogbn-arxiv': 40
    }

    def __init__(self, nfeat: int, dataset_name: str = None, nclass: int = None, nhid: int = 64, dropout: float = 0.5):
        super().__init__()

        # Determine number of classes
        if dataset_name and dataset_name.lower() in self.DATASET_CLASSES:
            nclass = self.DATASET_CLASSES[dataset_name.lower()]
        elif nclass is None:
            raise ValueError("Either dataset_name or nclass must be provided")

        self.gc1 = nn.Linear(nfeat, nhid)
        self.gc2 = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        # Ensure tensors are on the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)
        adj = adj.to(device)

        deg = adj.sum(dim=1, keepdim=True)
        deg[deg == 0] = 1
        adj_norm = adj / deg

        x = torch.mm(adj_norm, x)
        x = F.relu(self.gc1(x))
        x = self.dropout(x)

        x = torch.mm(adj_norm, x)
        x = self.gc2(x)
        return x
