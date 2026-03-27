import torch.nn as nn
import torch.nn.functional as F
import torch

from torch_geometric.nn import GATConv


class ResidualLinearBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_val):
        
        super().__init__()
        self.layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(dropout_val),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # Residual connection
        return x + self.layer(x)


class SkeletalMotionInterpolator(nn.Module):
    def __init__(self, context_len_pre, context_len_post, target_len, rot_gnn_params, root_pos_mlp_params):
        
        super().__init__()
        context_len = context_len_pre + context_len_post

        # GAT layers for rotations
        graph_in_features = rot_gnn_params["num_features_in"] * context_len
        hidden_dim = rot_gnn_params["hidden_dim"] 
        heads = rot_gnn_params["num_heads"]
        dropout_val = rot_gnn_params["dropout"]
        graph_out_features = hidden_dim * heads
        fc_rot_out_dim = rot_gnn_params["num_features_out"] * target_len
        
        self.convs = []
        self.convs.append(
            GATConv(
                in_channels=graph_in_features,
                out_channels=hidden_dim, 
                heads=heads, 
                concat=True, 
                dropout=dropout_val
            )
        )
        
        for _ in range(rot_gnn_params["num_layers"] - 1): 
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim * heads,
                    out_channels=hidden_dim, 
                    heads=heads, 
                    concat=True, 
                    dropout=dropout_val
                )
            )

        self.convs = nn.ModuleList(self.convs)
        self.dropout = nn.Dropout(dropout_val)
        self.fc_rot = nn.Linear(in_features=graph_out_features, out_features=fc_rot_out_dim)

        # MLP for root positions
        root_pos_in = (context_len * root_pos_mlp_params["num_features_in"]) + root_pos_mlp_params["num_features_in_from_gnn"]
        root_pos_out = target_len * root_pos_mlp_params["num_features_out"]
        hidden_dim = root_pos_mlp_params["hidden_dim"]
        dropout_val = root_pos_mlp_params["dropout"]

        self.reduce_graph_dim = nn.Linear(graph_out_features, root_pos_mlp_params["num_features_in_from_gnn"])
    
        mlp_layers = []
        mlp_layers.append(nn.Linear(in_features=root_pos_in, out_features=hidden_dim))

        for _ in range(root_pos_mlp_params["num_layers"] - 2):
            mlp_layers.append(ResidualLinearBlock(hidden_dim=hidden_dim, dropout_val=dropout_val))

        mlp_layers.append(nn.LeakyReLU())
        mlp_layers.append(nn.Linear(in_features=hidden_dim, out_features=root_pos_out))

        self.root_pos_mlp = nn.Sequential(*mlp_layers)


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
        if not hasattr(data, 'num_graphs'): batch_size = 1
        else: batch_size = data.num_graphs

        root_pos_ctx = data.root_pos_ctx.view(batch_size, -1) 
        J = data.num_nodes // batch_size
        graph_out_reshaped = x.view(batch_size, J, -1) # [B, J, graph_out_features]
        root_graph_out = graph_out_reshaped[:, 0, :]
        root_graph_reduced = F.relu(self.reduce_graph_dim(root_graph_out))
        root_mlp_ctx = torch.cat([root_pos_ctx, root_graph_reduced], dim=1) 

        root_pos_pred = self.root_pos_mlp(root_mlp_ctx) 

        return {'rot': rot_pred, 'root_pos': root_pos_pred}
