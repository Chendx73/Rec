import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu):
        super(GraphConvSparse, self).__init__()
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, x):
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


class GAE(nn.Module):
    def __init__(self, adj, args):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hd_dim1, adj)
        self.gcn_mean = GraphConvSparse(args.hd_dim1, args.hd_dim2, adj, activation=lambda x: x)
        self.mean = None
        self.args = args

    def encoder(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encoder(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class VGAE(GAE):
    def __init__(self, adj, args):
        super(VGAE, self).__init__(adj, args)
        self.gcn_logstddev = GraphConvSparse(args.hd_dim1, args.hd_dim2, adj, activation=lambda x: x)
        self.logstd = None

    def encoder(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), self.args.hd_dim2)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encoder(X)
        A_pred = dot_product_decode(Z)
        return A_pred
