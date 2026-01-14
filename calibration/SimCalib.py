import torch
import torch.nn.functional as F

'''
SimCalib is a post-hoc calibration method for Graph Neural Networks (GNNs)
that adjusts predictions based on the similarity between nodes.

Instead of applying a single global temperature, 
SimCalib computes a node-specific temperature by measuring how similar a test node is to confident validation nodes,
using both their features and predicted probabilities. 

More similar nodes contribute more to the calibration of a test node’s logits,
allowing SimCalib to capture local uncertainty patterns in the graph.

This leads to better-calibrated confidence estimates, especially in heterogeneous graphs.
'''
# SimCalib Implementation
class SimCalib(torch.nn.Module):
    def __init__(self, base_model, features, labels, adj, val_mask, k=10, epsilon=1e-8):
        super(SimCalib, self).__init__()
        self.base_model = base_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.epsilon = epsilon

        # Store data for initialization
        self.x = features.to(self.device)
        self.y = labels.to(self.device)
        self.adj = adj.to(self.device)
        self.val_mask = val_mask.to(self.device)

        # Move base model to device
        self.base_model = self.base_model.to(self.device)

        # Initialize similarity-based calibration
        self._initialize_calibration()

    def _initialize_calibration(self):
        """Initialize calibration by computing validation features and confidence."""
        # Set model to eval mode for feature extraction
        self.base_model.eval()

        with torch.no_grad():
            # Compute validation features and confidence
            self.features_val = self.latent_feature_1(self.x, self.adj)[self.val_mask]
            val_logits = self.base_model(self.x, self.adj)[self.val_mask]
            self.val_confidence = F.softmax(val_logits, dim=1).max(dim=1).values

    def compute_similarity_matrix(self, X1, X2):
        """Compute cosine similarity matrix efficiently using PyTorch."""
        # Normalize features
        X1_norm = F.normalize(X1, p=2, dim=1)
        X2_norm = F.normalize(X2, p=2, dim=1)

        # Compute cosine similarity using matrix multiplication
        similarity = torch.mm(X1_norm, X2_norm.t())

        return similarity

    def latent_feature_1(self, x, adj):
        """Extract features from the first layer of CompatibleGCN."""
        # Move tensors to device
        x, adj = x.to(self.device), adj.to(self.device)

        # CompatibleGCN handles normalization internally, so we replicate its first layer
        # Normalize adjacency matrix the same way as CompatibleGCN
        deg = adj.sum(dim=1, keepdim=True)
        deg[deg == 0] = 1
        adj_norm = adj / deg

        # Apply first graph convolution layer (without dropout during inference)
        x = torch.mm(adj_norm, x)
        x = F.relu(self.base_model.gc1(x))
        # Don't apply dropout during inference

        return x
    
    def forward(self, x, adj):
        x, adj = x.to(self.device), adj.to(self.device)

        # Set model to eval mode for base model parameters (but keep grad for adj)
        self.base_model.eval()

        # Get logits for all nodes (differentiable)
        all_logits = self.base_model(x, adj)

        # Get latent features for all nodes (differentiable)
        latent_features = self.latent_feature_1(x, adj)

        # Compute similarities between all nodes and validation nodes
        sim_matrix = self.compute_similarity_matrix(latent_features, self.features_val.detach())

        # Use soft top-k instead of hard top-k for differentiability
        # Apply softmax with temperature to create soft selection weights
        tau = 0.1  # temperature for soft selection (lower = sharper, higher = softer)

        # Create soft weights over validation nodes using temperature-scaled softmax
        soft_weights = F.softmax(sim_matrix / tau, dim=1)  # shape: [N_nodes, N_val]

        # Compute temperature for each node using soft-weighted validation confidences
        # Higher confidence validation nodes should lead to lower temperatures
        val_inv_conf = 1.0 / (self.val_confidence.detach() + self.epsilon)  # shape: [N_val]

        # Weighted sum: each node gets temperature from soft-weighted val nodes
        temperatures = torch.mm(soft_weights, val_inv_conf.unsqueeze(1)).squeeze(1)  # shape: [N_nodes]
        temperatures = temperatures.clamp(min=0.1, max=5.0)  # clipping for stability

        # Apply temperature scaling to all logits
        calibrated_logits = all_logits / temperatures.unsqueeze(1)

        return F.log_softmax(calibrated_logits, dim=1)

    def calib_train(self):
        """
        Training method for SimCalib (not needed as SimCalib is post-hoc).
        SimCalib doesn't require training as it uses similarity-based temperature scaling.
        """
        print("SimCalib is a post-hoc calibration method and doesn't require training.")
        print("Calibration is based on similarity computation, not learned parameters.")
        return self