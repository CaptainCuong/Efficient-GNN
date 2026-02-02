import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CoraFull, CitationFull, GNNBenchmarkDataset
from calib_attack.gcn import SRG_GCN
from calib_attack.calib_fga import Calib_FGA
from utils.ece import calculate_ece
from calib_attack.utils import edge_index_to_torch_matrix
from utils.save_data import save_list_to_csv
from utils.ece import ece_chart, ece_chart_one_class
from utils.data_help import preprocess_data

parser = argparse.ArgumentParser(description='Calibrated Graph Convolutional Networks')
parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
parser.add_argument('--calib-method', type=str, default='TS', choices=['TS', 'VS', 'CaGCN', 'MS', 'DCGC','GATS','WATS','SimCalib'], help='Calibration method')
parser.add_argument('--n-hidden', type=int, default=16, help='Number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
parser.add_argument('--epochs', type=int, default=250, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--attack-strategy', type=str, default='under', help='Attack strategy: over | under | target | test')
parser.add_argument('--target-label', type=int, default=3, help='Specific label to target for attack')

args = parser.parse_args()

################## Dataset/Model Preparation ##################

# Load the dataset
if args.dataset == 'citeseer':
    dataset = Planetoid(root="~/datasets/gnn/", name="CiteSeer")
elif args.dataset == 'cora':
    dataset = Planetoid(root="~/datasets/gnn/", name="Cora")
elif args.dataset == 'pubmed':
    dataset = Planetoid("~/datasets/gnn", name="PubMed")
elif args.dataset == 'dblp':
    dataset = CitationFull("~/datasets/gnn", "dblp")
    preprocess_data(dataset.data, 0.125, 0.125)
elif args.dataset == 'cora_ml':
    dataset = CitationFull("~/datasets/gnn", "cora_ml")
    preprocess_data(dataset.data, 0.125, 0.125)
elif args.dataset == 'pattern':
    dataset = GNNBenchmarkDataset("~/datasets/gnn", "PATTERN") # name in ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']
    preprocess_data(dataset.data, 0.125, 0.125)
elif args.dataset == 'cluster':
    dataset = GNNBenchmarkDataset("~/datasets/gnn", "PATTERN") # name in ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']
    preprocess_data(dataset.data, 0.125, 0.125)
elif args.dataset == 'mnist':
    dataset = GNNBenchmarkDataset("~/datasets/gnn", "MNIST") # name in ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']
    preprocess_data(dataset.data, 0.125, 0.125)
elif args.dataset == 'cifar10':
    dataset = GNNBenchmarkDataset("~/datasets/gnn", "CIFAR10") # name in ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']
    preprocess_data(dataset.data, 0.125, 0.125)
elif args.dataset == 'tsp':
    dataset = GNNBenchmarkDataset("~/datasets/gnn", "TSP") # name in ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']
    preprocess_data(dataset.data, 0.125, 0.125)
elif args.dataset == 'csl':
    dataset = GNNBenchmarkDataset("~/datasets/gnn", "CSL") # name in ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']
    preprocess_data(dataset.data, 0.125, 0.125)
else:
    raise ValueError("Invalid dataset name. Please choose from 'citeseer' or 'cora'.")

data = dataset.data
n_classes = dataset.num_classes

# Extract masks
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# Further split validation set into validation and calibration sets
val_indices = val_mask.nonzero(as_tuple=True)[0]
calibration_size = int(0.5 * len(val_indices))
calibration_indices = val_indices[:calibration_size]
new_val_indices = val_indices[calibration_size:]

# Update masks
# Calibrated or Validated Nodes have value True
calibration_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
calibration_mask[calibration_indices] = True

val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask[new_val_indices] = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################## Training Model ##################
print("Start training the model...")
adj_sqr = edge_index_to_torch_matrix(data.edge_index.cpu(), data.x.shape[0])

model = SRG_GCN(nfeat=data.x.shape[1], nclass=n_classes,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
train_indices = train_mask.nonzero(as_tuple=True)[0]
model.fit(data.x, adj_sqr, data.y, train_indices, new_val_indices, patience=30)
# Save the trained model
# torch.save(model.state_dict(), './saved_models/srg_gcn_model.pth')

from calib_models.TS import TemperatureScaling
from calib_models.MS import MatrixScaling
from calib_models.VS import VectorScaling
from calib_models.DCGC import DCGC
from calib_models.ISO import ISO
from calib_models.GATS import GATS
from calib_models.CaGCN import CaGCN
from calib_models.WATS import WATS
from calib_models.SimCalib import SimCalib

if args.calib_method == 'TS':
    calib_model = TemperatureScaling(n_classes, model, data.x, data.y, adj_sqr, val_indices)
elif args.calib_method == 'VS':
    calib_model = VectorScaling(n_classes, model, data.x, data.y, adj_sqr, val_indices)
elif args.calib_method == 'MS':
    calib_model = MatrixScaling(n_classes, model, data.x, data.y, adj_sqr, val_indices)
elif args.calib_method == 'DCGC':
    calib_model = DCGC(n_classes, model, data.x, data.y, adj_sqr, val_indices)
elif args.calib_method == 'ISO':
    calib_model = ISO(model, data.x, data.y, adj_sqr, n_classes, val_indices)
elif args.calib_method == 'GATS':
    num_nodes = data.x.shape[0]
    calib_model = GATS(model, adj_sqr, num_nodes, val_indices, n_classes)
elif args.calib_method == 'CaGCN':
    calib_model = CaGCN(n_classes, model, data.x, data.y, adj_sqr, val_indices)
elif args.calib_method == 'WATS':
    calib_model = WATS(model, data.x, data.y, adj_sqr, calibration_mask)
elif args.calib_method == 'SimCalib':
    calib_model = SimCalib(model, data.x, adj_sqr, calibration_mask)

################## Adversarial Graph Generation ##################
print("Start generating adversarial graphs...")

# Evaluate test set
test_indices = test_mask.nonzero(as_tuple=True)[0]
# test_indices = torch.where((test_mask == True) & (data.y == 3))[0]
test_labels = data.y[test_indices].cpu().numpy()

origin_probs = F.softmax(calib_model(data.x, adj_sqr, test_idx=test_indices).detach()[test_indices], dim=1)
att_probs = torch.empty(len(test_indices), n_classes)

best_conf = []
n_perturb_con = []
for idx, target_node in enumerate(test_indices):
    target_node = target_node.item()
    attacker = Calib_FGA(calib_model, attack_structure=True, attack_features=False, device=device).to(device)
    attacker.rerank_hybridloss_attack(data.x, adj_sqr, target_node, 5, args.attack_strategy, target_label=args.target_label, res_gt=data.y, best_conf=best_conf, n_perturb=n_perturb_con)
    modified_adj = attacker.modified_adj
    with torch.no_grad():
        modified_probs = F.softmax(calib_model(data.x.to(device), modified_adj, test_idx=target_node)[target_node], dim=0)
    att_probs[idx] = modified_probs.to("cpu")
    torch.cuda.empty_cache()

file_name = f"rerank_hybridloss"
data_dir = os.path.expanduser('~/data')
os.makedirs(data_dir, exist_ok=True)
np.save(os.path.join(data_dir, f'{file_name}_{args.dataset}_{args.calib_method}_test_labels.npy'), test_labels)
np.save(os.path.join(data_dir, f'{file_name}_{args.dataset}_{args.calib_method}_org_probs.npy'), np.array(origin_probs.cpu()))
np.save(os.path.join(data_dir, f'{file_name}_{args.dataset}_{args.calib_method}_att_probs.npy'), np.array(att_probs.detach()))
np.save(os.path.join(data_dir, f'{file_name}_{args.dataset}_{args.calib_method}_best_conf.npy'), np.array(best_conf))
np.save(os.path.join(data_dir, f'{file_name}_{args.dataset}_{args.calib_method}_n_perturb_con.npy'), np.array(n_perturb_con))