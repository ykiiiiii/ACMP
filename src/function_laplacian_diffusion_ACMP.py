import torch
from torch import nn
import torch_sparse

from base_classes import ODEFunc
from utils import MaxNFEException


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(LaplacianODEFunc, self).__init__(opt, data, device)
    print('using ACMP')
    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))
    self.particle_beta = torch.tensor(opt['beta']).to(device)
  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      edge_weight = mean_attention - self.particle_beta
      ax = torch_sparse.spmm(self.edge_index, edge_weight, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      edge_weight = (self.attention_weights - self.particle_beta)
      ax = torch_sparse.spmm(self.edge_index, edge_weight, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      edge_weight = (self.edge_weight - self.particle_beta)
      ax = torch_sparse.spmm(self.edge_index, edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    ax = self.sparse_multiply(x)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
      delta = torch.sigmoid(self.delta_train)
    else:
      alpha = self.alpha_train
      delta = self.delta_train

    diffusion = ax - x
    if self.opt['channel_mixing']:
      f = -delta*(x**2-self.opt['barrier'])*x + diffusion@alpha
    else:
      f = -delta*(x**2-self.opt['barrier'])*x + alpha*diffusion

    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f
