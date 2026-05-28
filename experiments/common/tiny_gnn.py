import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool


class TinyGNN(nn.Module):
    def __init__(
        self,
        in_dim=1,          # node feature dim 
        edge_dim=1,        # edge feature dim 
        hidden=64,
        depth=2,           # number of GINE layers 
        dropout=0.1,
        pool="mean",       
        use_bn=True,       
        residual=True
    ):
        super().__init__()
        self.dropout = dropout
        self.use_bn = use_bn
        self.residual = residual
        self.pool = pool

        # Node input projection
        self.in_proj = nn.Linear(in_dim, hidden)
        self.edge_lin = nn.Linear(edge_dim, edge_dim)

        # Build GINE blocks
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(depth):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            conv = GINEConv(mlp, edge_dim=edge_dim, train_eps=True)
            self.convs.append(conv)
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden))

        # Head 
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden // 2, 1),
        )

    def _pool(self, h, batch):
        if self.pool == "add":
            return global_add_pool(h, batch)
        elif self.pool == "max":
            return global_max_pool(h, batch)
        else:
            return global_mean_pool(h, batch)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if x.dim() == 1:
            x = x.view(-1, 1)
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()

        # Project
        h = self.in_proj(x)
        e = self.edge_lin(edge_attr) if edge_attr is not None else None

        # Convs
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index, e)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual and h_in.shape == h.shape:
                h = h + h_in

        # Pool → head
        g = self._pool(h, batch)
        yhat = self.head(g).view(-1)
        return yhat, g