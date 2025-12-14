import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv


class SkeletalMotionInterpolator(nn.Module):
    def __init__(self, context_len_pre, context_len_post, target_len, hidden_dim, hidden_layers, root_pos_hidden_dim,
            heads, dropout, node_features, graph_features):
        
        super().__init__()
        context_len = context_len_pre + context_len_post

        # GAT layers for rotations
        graph_in_features = node_features * context_len
        graph_out_features = node_features * target_len

        self.convs = []
        self.convs.append(GATConv(graph_in_features, hidden_dim, heads=heads, concat=True, dropout=dropout))
        
        for _ in range(hidden_layers - 1): 
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout))

        self.convs = nn.ModuleList(self.convs)
        self.dropout = nn.Dropout(dropout)
        self.fc_rot = nn.Linear(hidden_dim * heads, graph_out_features)

        # MLP for root positions
        root_pos_in = context_len * graph_features
        root_pos_out = target_len * graph_features

        self.root_pos_mlp = nn.Sequential(
            nn.Linear(root_pos_in, root_pos_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(root_pos_hidden_dim, root_pos_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(root_pos_hidden_dim, root_pos_out)
        )

    def forward(self, data):
        # Rotations
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.leaky_relu(x)
                x = self.dropout(x)

        rot_pred = self.fc_rot(x) 

        # Root positions
        root_pos_ctx = data.root_pos_ctx
        if not hasattr(data, 'num_graphs'): root_pos_ctx = root_pos_ctx.reshape(1, -1)
        else: root_pos_ctx = root_pos_ctx.view(data.num_graphs, -1) 
        root_pos_pred = self.root_pos_mlp(root_pos_ctx) 

        return {'rot': rot_pred, 'root_pos': root_pos_pred}
