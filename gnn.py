import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class ReqMessagePassing(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_attr_dim, req_dim):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + edge_attr_dim + req_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, req_emb):
        if req_emb.dim() == 1:
            req_emb = req_emb.view(1, -1)
        if req_emb.size(0) == 1:
            req_emb = req_emb.expand(edge_index.size(1), -1)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, req_emb=req_emb)

    def message(self, x_j, edge_attr, req_emb):
        msg_input = torch.cat([x_j, edge_attr, req_emb], dim=-1)
        return self.mlp(msg_input)


class DirGINELayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_attr_dim, req_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_attr_dim = edge_attr_dim
        self.req_dim = req_dim

        self.msg_in = ReqMessagePassing(in_dim, out_dim, edge_attr_dim, req_dim)
        self.msg_out = ReqMessagePassing(in_dim, out_dim, edge_attr_dim, req_dim)

        self.lin_self = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.eps = nn.Parameter(torch.zeros(1))
        self.update_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, edge_attr, req_emb):
        num_edges = edge_index.size(1)
        x_self = self.lin_self(x)
        if num_edges == 0:
            h = (1.0 + self.eps) * x_self
            return self.update_mlp(h)
        
        assert num_edges % 2 == 0, "edge_index 应该是 forward + backward 成对构造的"
        E = num_edges // 2

        edge_index_f = edge_index[:, :E]
        edge_index_b = edge_index[:, E:]
        edge_attr_f = edge_attr[:E]
        edge_attr_b = edge_attr[E:]

        h_in = self.msg_in(x, edge_index_f, edge_attr_f, req_emb)
        h_out = self.msg_out(x, edge_index_b, edge_attr_b, req_emb)
        h = (1.0 + self.eps) * x_self + h_in + h_out
        return self.update_mlp(h)


class GNNEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 dropout=0.0,
                 edge_attr_dim=0,
                 req_dim=0):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.edge_attr_dim = edge_attr_dim
        self.req_dim = req_dim

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layer = DirGINELayer(
                in_dim=in_dim,
                out_dim=hidden_dim,
                edge_attr_dim=edge_attr_dim, 
                req_dim=req_dim,
                dropout=dropout
            )
            layers.append(layer)
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

        if output_dim != hidden_dim:
            self.lin_out = nn.Linear(hidden_dim, output_dim)
        else:
            self.lin_out = nn.Identity()
    
    def forward(self, x, edge_index, edge_attr, req_emb):
        if edge_attr is None:
            num_edges = edge_index.size(1)
            edge_attr = x.new_zeros(num_edges, self.edge_attr_dim)

        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, req_emb)

        h = self.lin_out(h)
        return h