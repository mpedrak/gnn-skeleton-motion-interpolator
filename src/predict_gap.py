import torch
import numpy as np

from torch_geometric.data import Data

from .utils.bvh import build_edge_index_from_parents


@torch.no_grad()
def predict_gap(model, device, rot_6d, root_pos, parent_indices, context_len_pre, context_len_post, target_len, gap_start):
    
    J = rot_6d.shape[1]
    second_start = gap_start + target_len
    end = second_start + context_len_post
    first_start = gap_start - context_len_pre

    # Rotations
    first_part_rot = rot_6d[first_start : gap_start]
    second_part_rot = rot_6d[second_start : end]
    rot_ctx = np.concatenate([first_part_rot, second_part_rot], axis=0)
    x_feat = torch.tensor(rot_ctx, dtype=torch.float32).permute(1, 0, 2).reshape(J, -1) # [J, F * 6]

    # Root positions
    first_part_root_pos = root_pos[first_start : gap_start]
    second_part_root_pos = root_pos[second_start : end] 
    first_ctx_root_pos = first_part_root_pos[0].clone()
    first_part_root_pos = first_part_root_pos - first_ctx_root_pos
    second_part_root_pos = second_part_root_pos - first_ctx_root_pos 
    root_pos_ctx = torch.cat([first_part_root_pos, second_part_root_pos], dim=0).to(device) 

    # Graph
    edge_index = build_edge_index_from_parents(parent_indices)

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        root_pos_ctx=root_pos_ctx
    )
    data = data.to(device)
    out = model(data)
    
    # Reshape rotations
    rot_pred = out["rot"]
    rot_pred = rot_pred.view(J, target_len, 6).permute(1, 0, 2).contiguous() # [J, F, 6] -> [F, J, 6]      

    # Reconstruct root positions from deltas
    root_pos_delta_pred = out["root_pos"]
    root_pos_delta_pred = root_pos_delta_pred.view(1, -1).view(target_len, 3)
    root_pos_pred = root_pos_delta_pred.to(device) + first_ctx_root_pos.to(device)            

    return rot_pred.cpu(), root_pos_pred.cpu()
