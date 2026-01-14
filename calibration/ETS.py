import torch
from torch import nn
from .TS import TemperatureScaling as TS
import torch.nn.functional as F
import numpy as np
import scipy.optimize

class ETS(nn.Module):
    def __init__(self, model, x, y, adj, val_idx):
        super().__init__()
        self.model = model
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.zeros(1))
        self.w3 = nn.Parameter(torch.zeros(1))
        self.num_classes = y.max().item() + 1  # Calculate n_classes from labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_model = TS(model, x, y, adj, val_idx)
        self.x = x
        self.y = y
        self.adj = adj
        self.val_idx = val_idx
    def forward(self, x, adj):
        logits = self.model(x, adj)
        temp = torch.log(torch.exp(self.temp_model.temperature) + torch.tensor(1.1)).to(self.device)
        p = self.w1 * F.softmax(logits / temp, dim=1) + self.w2 * F.softmax(logits, dim=1) + self.w3 * 1/self.num_classes
        return torch.log(p)

    def fit(self, masks):
        self.to(self.device)
        # Temperature scaling is already fitted during temp_model initialization
        torch.cuda.empty_cache()

        # Ensure all tensors are on the same device
        x = self.x.to(self.device)
        adj = self.adj.to(self.device)
        y = self.y.to(self.device)
        val_mask = masks[1].to(self.device)

        logits = self.model(x, adj)[val_mask]
        label = y[val_mask]
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.unsqueeze(-1), 1)
        temp = self.temp_model.temperature.cpu().detach().numpy()
        w = self.ensemble_scaling(logits.cpu().detach().numpy(), one_hot.cpu().detach().numpy(), temp)
        self.w1.data = torch.tensor(w[0], device=self.device)
        self.w2.data = torch.tensor(w[1], device=self.device)
        self.w3.data = torch.tensor(w[2], device=self.device)
        return self

    def ensemble_scaling(self, logit, label, t):
        """
        Official ETS implementation from Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning
        Code taken from (https://github.com/zhang64-llnl/Mix-n-Match-Calibration)
        Use the scipy optimization because PyTorch does not have constrained optimization.
        """
        p1 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        logit = logit/t
        p0 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        p2 = np.ones_like(p0)/self.num_classes
        

        bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
        def my_constraint_fun(x): return np.sum(x)-1
        constraints = { "type":"eq", "fun":my_constraint_fun,}
        w = scipy.optimize.minimize(ETS.ll_w, (1.0, 0.0, 0.0), args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': False})
        w = w.x
        return w

    @staticmethod
    def ll_w(w, *args):
    ## find optimal weight coefficients with Cros-Entropy loss function
        p0, p1, p2, label = args
        p = (w[0]*p0+w[1]*p1+w[2]*p2)
        N = p.shape[0]
        ce = -np.sum(label*np.log(p))/N
        return ce   