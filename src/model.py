import torch.nn as nn
import torch.nn.functional as F
import torch

from torch_geometric.nn import GATConv, global_mean_pool


class SkeletalMotionInterpolator(nn.Module):
    def __init__(self, context_len_pre, context_len_post, target_len, hidden_dim, hidden_layers, root_pos_hidden_dim,
            heads, dropout, node_features, graph_features, num_joints):
        
        super().__init__()
        self.context_len = context_len_pre + context_len_post
        self.graph_features = graph_features
        self.num_joints = num_joints
        self.node_features = node_features
        self.target_len = target_len

        in_features = node_features

        self.convs = []
        self.convs.append(GATConv(in_features, hidden_dim, heads=heads, concat=True, dropout=dropout))
        
        for _ in range(hidden_layers - 1): 
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout))

        self.convs = torch.nn.ModuleList(self.convs)
        self.dropout = nn.Dropout(dropout)
        
        rot_out_dim = self.num_joints * self.target_len * node_features
        self.fc_rot = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim * heads),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * heads, rot_out_dim),
        )

        root_in = self.context_len * graph_features
        root_out = target_len * graph_features

        self.root_head = nn.Sequential(
            nn.Linear(root_in, root_pos_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(root_pos_hidden_dim, root_pos_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(root_pos_hidden_dim, root_out)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.leaky_relu(x)
                x = self.dropout(x)

        x = global_mean_pool(x, data.batch)
        B = x.size(0)

        rot_pred = self.fc_rot(x) 
        rot_pred = rot_pred.view(B, self.num_joints, self.target_len * self.node_features)

        root_ctx_norm = data.root_ctx_norm  
        if root_ctx_norm.dim() != 1: root_ctx_norm = root_ctx_norm.view(-1)

        total_len = len(root_ctx_norm)
        per_graph = self.context_len * self.graph_features
        batch_size = total_len // per_graph
        root_ctx_norm = root_ctx_norm.view(batch_size, per_graph)
       
        root_norm_pred = self.root_head(root_ctx_norm) 

        return {'rot': rot_pred, 'root_norm': root_norm_pred}
    