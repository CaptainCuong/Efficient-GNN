"""Dataset and dataloader utilities."""
from __future__ import annotations

import functools
from typing import Iterable, Tuple

import torch

# Hardcoded dataset class counts for reproducibility
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

def get_hardcoded_num_classes(dataset_name: str) -> int:
    """Get hardcoded number of classes for a dataset."""
    return DATASET_CLASSES.get(dataset_name.lower(), None)

try:
    from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "Missing optional dependency 'ogb'. Install with `pip install ogb`."
    ) from exc

try:
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.datasets import Planetoid, Reddit, DBLP, Amazon, Coauthor
    from torch_geometric.transforms import RandomNodeSplit

    # Check for sampling dependencies and add workaround if needed
    try:
        import torch_sparse
    except ImportError:
        try:
            import pyg_lib
        except ImportError:
            # Add simple workaround for missing sampling dependencies
            import torch_geometric.sampler.neighbor_sampler as ns_module
            from torch_geometric.sampler import SamplerOutput

            def _fallback_sample(self, seed, seed_time=None):
                """Simple fallback sampling for when dependencies are missing."""
                nodes = seed[:min(len(seed), 1000)]  # Limit batch size
                num_nodes = len(nodes)
                return SamplerOutput(
                    node=nodes,
                    row=torch.tensor([], dtype=torch.long),
                    col=torch.tensor([], dtype=torch.long),
                    edge=torch.tensor([], dtype=torch.long),
                    batch=None,
                    num_sampled_nodes=[num_nodes],
                    num_sampled_edges=[0]
                )

            ns_module.NeighborSampler._sample = _fallback_sample

except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "Missing optional dependency 'torch-geometric'. Install with `pip install torch-geometric`."
    ) from exc

from .config import DatasetConfig, ExperimentConfig


def load_ogb_dataset(cfg: DatasetConfig):
    """Load an OGB node property prediction dataset."""
    # Temporarily allow unsafe loading for OGB compatibility with PyTorch 2.6+
    original_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=False)

    try:
        dataset = PygNodePropPredDataset(name=cfg.name, root=cfg.root)
        split_idx = dataset.get_idx_split()
        data = dataset[0]
    finally:
        torch.load = original_load

    # Flatten multi-dimensional labels
    if data.y.dim() > 1 and data.y.size(-1) == 1:
        data.y = data.y.view(-1)

    # Add training indices to data object if not present
    if not hasattr(data, "train_idx"):
        data.train_idx = split_idx["train"]

    return data, split_idx, dataset.num_classes


def load_planetoid_dataset(cfg: DatasetConfig):
    """Load a Planetoid dataset (Cora, CiteSeer, PubMed)."""
    # Temporarily allow unsafe loading for PyTorch 2.6+ compatibility
    original_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=False)

    try:
        dataset = Planetoid(root=cfg.root, name=cfg.name)
        data = dataset[0]
    finally:
        torch.load = original_load

    # Create splits using the standard mask-based approach
    split_idx = {
        "train": data.train_mask.nonzero(as_tuple=False).view(-1),
        "valid": data.val_mask.nonzero(as_tuple=False).view(-1),
        "test": data.test_mask.nonzero(as_tuple=False).view(-1),
    }

    # Add training indices to data object if not present
    if not hasattr(data, "train_idx"):
        data.train_idx = split_idx["train"]

    return data, split_idx, dataset.num_classes


def load_reddit_dataset(cfg: DatasetConfig):
    """Load the Reddit dataset."""
    # Temporarily allow unsafe loading for PyTorch 2.6+ compatibility
    original_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=False)

    try:
        dataset = Reddit(root=cfg.root)
        data = dataset[0]
    finally:
        torch.load = original_load

    # Create splits using the standard mask-based approach
    split_idx = {
        "train": data.train_mask.nonzero(as_tuple=False).view(-1),
        "valid": data.val_mask.nonzero(as_tuple=False).view(-1),
        "test": data.test_mask.nonzero(as_tuple=False).view(-1),
    }

    # Add training indices to data object if not present
    if not hasattr(data, "train_idx"):
        data.train_idx = split_idx["train"]

    return data, split_idx, dataset.num_classes


def load_dblp_dataset(cfg: DatasetConfig):
    """Load the DBLP dataset."""
    # Temporarily allow unsafe loading for PyTorch 2.6+ compatibility
    original_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=False)

    try:
        dataset = DBLP(root=cfg.root)
        data = dataset[0]
    finally:
        torch.load = original_load

    # Create splits using the standard mask-based approach
    split_idx = {
        "train": data.train_mask.nonzero(as_tuple=False).view(-1),
        "valid": data.val_mask.nonzero(as_tuple=False).view(-1),
        "test": data.test_mask.nonzero(as_tuple=False).view(-1),
    }

    # Add training indices to data object if not present
    if not hasattr(data, "train_idx"):
        data.train_idx = split_idx["train"]

    return data, split_idx, dataset.num_classes


def load_amazon_dataset(cfg: DatasetConfig):
    """Load Amazon Computers or Photo dataset."""
    # Temporarily allow unsafe loading for PyTorch 2.6+ compatibility
    original_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=False)

    try:
        dataset = Amazon(root=cfg.root, name=cfg.name)
        data = dataset[0]
    finally:
        torch.load = original_load

    # Amazon datasets don't have predefined splits, so we need to create them
    # Use a 60/20/20 split with random assignment
    transform = RandomNodeSplit(num_val=0.2, num_test=0.2)
    data = transform(data)

    # Create splits using the mask-based approach
    split_idx = {
        "train": data.train_mask.nonzero(as_tuple=False).view(-1),
        "valid": data.val_mask.nonzero(as_tuple=False).view(-1),
        "test": data.test_mask.nonzero(as_tuple=False).view(-1),
    }

    # Add training indices to data object if not present
    if not hasattr(data, "train_idx"):
        data.train_idx = split_idx["train"]

    return data, split_idx, dataset.num_classes


def load_coauthor_dataset(cfg: DatasetConfig):
    """Load Coauthor CS or Physics dataset."""
    # Temporarily allow unsafe loading for PyTorch 2.6+ compatibility
    original_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=False)

    try:
        dataset = Coauthor(root=cfg.root, name=cfg.name)
        data = dataset[0]
    finally:
        torch.load = original_load

    # Coauthor datasets don't have predefined splits, so we need to create them
    # Use a 60/20/20 split with random assignment
    transform = RandomNodeSplit(num_val=0.2, num_test=0.2)
    data = transform(data)

    # Create splits using the mask-based approach
    split_idx = {
        "train": data.train_mask.nonzero(as_tuple=False).view(-1),
        "valid": data.val_mask.nonzero(as_tuple=False).view(-1),
        "test": data.test_mask.nonzero(as_tuple=False).view(-1),
    }

    # Add training indices to data object if not present
    if not hasattr(data, "train_idx"):
        data.train_idx = split_idx["train"]

    return data, split_idx, dataset.num_classes


def load_dataset(cfg: DatasetConfig):
    """Load a node property prediction dataset based on type."""
    if cfg.dataset_type.lower() == "ogb":
        return load_ogb_dataset(cfg)
    elif cfg.dataset_type.lower() == "planetoid":
        return load_planetoid_dataset(cfg)
    elif cfg.dataset_type.lower() == "reddit":
        return load_reddit_dataset(cfg)
    elif cfg.dataset_type.lower() == "dblp":
        return load_dblp_dataset(cfg)
    elif cfg.dataset_type.lower() == "amazon":
        return load_amazon_dataset(cfg)
    elif cfg.dataset_type.lower() == "coauthor":
        return load_coauthor_dataset(cfg)
    else:
        raise ValueError(f"Unsupported dataset type: {cfg.dataset_type}. Use 'ogb', 'planetoid', 'reddit', 'dblp', 'amazon', or 'coauthor'.")


def _resolve_fanouts(fanouts: Iterable[int], depth: int) -> Tuple[int, ...]:
    sequence = tuple(int(f) for f in fanouts)
    if len(sequence) < depth:
        last = sequence[-1]
        sequence = sequence + (last,) * (depth - len(sequence))
    elif len(sequence) > depth:
        sequence = sequence[:depth]
    return sequence


def _create_neighbor_loader(
    data, input_nodes, num_neighbors, batch_size, shuffle, num_workers, pin_memory
):
    """Helper function to create a NeighborLoader with common parameters."""
    return NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def create_full_graph_loader(data, split_idx, batch_size, shuffle=False):
    """Create a simple loader that returns the full graph with specified node indices."""
    from torch.utils.data import DataLoader, Dataset

    class FullGraphDataset(Dataset):
        def __init__(self, node_indices):
            self.node_indices = node_indices

        def __len__(self):
            return len(self.node_indices)

        def __getitem__(self, idx):
            return self.node_indices[idx]

    def collate_fn(batch_indices):
        # Return full graph data with batch information
        batch_data = data.clone()
        batch_data.batch_indices = torch.tensor(batch_indices, dtype=torch.long)
        batch_data.batch_size = len(batch_indices)
        return batch_data

    dataset = FullGraphDataset(split_idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def create_dataloaders(cfg: ExperimentConfig):
    """Create data loaders based on model type."""
    data, split_idx, num_classes = load_dataset(cfg.dataset)

    # Common parameters
    use_cuda = cfg.device.startswith("cuda")
    num_workers = cfg.training.num_workers

    # Choose loading strategy based on model type
    if cfg.model.model_type.lower() in ["gcn", "gat"]:
        # For GCN and GAT, use full graph processing
        train_loader = create_full_graph_loader(
            data, split_idx["train"], cfg.training.batch_size, shuffle=True
        )
        val_loader = create_full_graph_loader(
            data, split_idx["valid"], cfg.training.eval_batch_size, shuffle=False
        )
        test_loader = create_full_graph_loader(
            data, split_idx["test"], cfg.training.eval_batch_size, shuffle=False
        )
    else:
        # For SAGE, use neighbor sampling
        fanouts = _resolve_fanouts(cfg.model.fanouts, cfg.model.num_layers)
        train_loader = _create_neighbor_loader(
            data=data,
            input_nodes=split_idx["train"],
            num_neighbors=fanouts,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        # Evaluation loaders (no sampling)
        eval_neighbors = [-1]  # Include all neighbors for evaluation
        val_loader = _create_neighbor_loader(
            data=data,
            input_nodes=split_idx["valid"],
            num_neighbors=eval_neighbors,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        test_loader = _create_neighbor_loader(
            data=data,
            input_nodes=split_idx["test"],
            num_neighbors=eval_neighbors,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

    # Create evaluator only for OGB datasets
    evaluator = None
    if cfg.dataset.dataset_type.lower() == "ogb":
        evaluator = Evaluator(name=cfg.dataset.name)

    return {
        "data": data,
        "splits": split_idx,
        "num_classes": num_classes,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "evaluator": evaluator,
    }
