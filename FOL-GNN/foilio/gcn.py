import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import LayerNorm
import numpy as np



class GCN(nn.Module):
    def __init__(self,args):
        super().__init__()
        # self.gnn_projections = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(M)])
        # self.gnn_projections_ln = nn.ModuleList([LayerNorm(n_hidden) for _ in range(M)])
        self.n_hidden = args.bert_embedding_dim
        self.gnn_projections = nn.ModuleList([nn.Linear(self.n_hidden, self.n_hidden) for _ in range(args.gcn_num)])
        self.gnn_projections_ln = nn.ModuleList([LayerNorm(self.n_hidden) for _ in range(args.gcn_num)])

    @staticmethod
    def normalize_adj(adj, add_self_loop=True):
        if add_self_loop:
            idx = torch.arange(adj.size(1), dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

    def message_passing(self, hidden, function, adj):
        #hidden = torch.LongTensor(hidden)
        out = function(hidden)
        adj = self.normalize_adj(adj.float())
        return F.gelu(torch.matmul(adj.type_as(out), out))

    def aggregation(self, hidden, adj):
        for gnn_projection_ln, gnn_projection in zip(self.gnn_projections_ln, self.gnn_projections):
            hidden = self.message_passing(hidden, gnn_projection,adj)
            hidden = gnn_projection_ln(hidden)
        return hidden      


    def forward(self, seq,seq_mask,adj):
       
        representations = self.aggregation(seq, adj)
        #print(representations)


        #hidden = representations.masked_select(mask_idx)
        # hidden = representations.masked_select(mask_idx.unsqueeze(2).expand_as(representations)).view(
        #      -1, self.n_hidden)
        # print("hidden",hidden.shape)

        return representations
        #return representations

