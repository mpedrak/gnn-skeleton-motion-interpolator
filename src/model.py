import torch.nn as nn
import torch.nn.functional as F
import torch

from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import AttentionalAggregation


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
        gat_hidden_dim = rot_gnn_params["hidden_dim"] 
        heads = rot_gnn_params["num_heads"] 
        gat_out_features = gat_hidden_dim * heads
        
        self.convs = []
        self.convs.append(
            GATConv(
                in_channels=rot_gnn_params["num_features_in"] * context_len,
                out_channels=gat_hidden_dim, 
                heads=heads, 
                concat=True, 
                dropout=rot_gnn_params["dropout"]
            )
        )
        
        for _ in range(rot_gnn_params["num_layers"] - 1): 
            self.convs.append(
                GATConv(
                    in_channels=gat_hidden_dim * heads,
                    out_channels=gat_hidden_dim, 
                    heads=heads, 
                    concat=True, 
                    dropout=rot_gnn_params["dropout"]
                )
            )

        self.convs = nn.ModuleList(self.convs)
        self.gat_norm = nn.LayerNorm(gat_out_features)
        self.dropout = nn.Dropout(rot_gnn_params["dropout"])
        self.fc_rot = nn.Linear(in_features=gat_out_features, out_features=rot_gnn_params["num_features_out"] * target_len)

        # MLP for root positions
        root_pos_in = (context_len * root_pos_mlp_params["num_features_in"]) + root_pos_mlp_params["num_features_in_from_gnn"]
        root_pos_out = target_len * root_pos_mlp_params["num_features_out"]
        mlp_hidden_dim = root_pos_mlp_params["hidden_dim"]

        self.attention_net = nn.Sequential(
            nn.Linear(gat_out_features, mlp_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )
        self.global_pool = AttentionalAggregation(gate_nn=self.attention_net)

        self.reduce_graph_dim = nn.Linear(gat_out_features, root_pos_mlp_params["num_features_in_from_gnn"])
    
        mlp_layers = []
        mlp_layers.append(nn.Linear(in_features=root_pos_in, out_features=mlp_hidden_dim))

        for _ in range(root_pos_mlp_params["num_layers"] - 2):
            mlp_layers.append(ResidualLinearBlock(hidden_dim=mlp_hidden_dim, dropout_val=root_pos_mlp_params["dropout"]))

        mlp_layers.append(nn.LeakyReLU())
        mlp_layers.append(nn.Linear(in_features=mlp_hidden_dim, out_features=root_pos_out))

        self.root_pos_mlp = nn.Sequential(*mlp_layers)


    def forward(self, data):
        
        if not hasattr(data, 'num_graphs'): batch_size = 1
        else: batch_size = data.num_graphs

        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch') and data.batch is not None: batch = data.batch 
        else: batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Rotations
        for i, conv in enumerate(self.convs):
            x_prev = x
            x = conv(x, edge_index)
            if i != 0: x = x + x_prev # Residual connection

            if i != len(self.convs) - 1:
                x = self.gat_norm(x)
                x = F.leaky_relu(x)
                x = self.dropout(x)     

        rot_pred = self.fc_rot(x) 

        # Root positions
        root_pos_ctx = data.root_pos_ctx.view(batch_size, -1) 
        root_graph_out = self.global_pool(x, batch)
        root_graph_reduced = F.leaky_relu(self.reduce_graph_dim(root_graph_out))
        root_mlp_ctx = torch.cat([root_pos_ctx, root_graph_reduced], dim=1) 

        root_pos_pred = self.root_pos_mlp(root_mlp_ctx) 

        return {'rot': rot_pred, 'root_pos': root_pos_pred}
