import torch
import numpy as np

from torch_geometric.data import Data

from .utils.bvh import compute_root_deltas, build_edge_index_from_parents


@torch.no_grad()
def predict_gap(model, device, rot_6d, root_pos, parent_indices, context_len_pre, context_len_post, target_len, gap_start, 
        root_mean, root_std):
    
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
    first_part_root_pos = compute_root_deltas(first_part_root_pos)
    second_part_root_pos = root_pos[second_start : end]
    second_part_root_pos = compute_root_deltas(second_part_root_pos)
    
    root_ctx_pos = torch.cat([first_part_root_pos, second_part_root_pos], dim=0).to(device) 
    root_ctx_norm = (root_ctx_pos - root_mean) / root_std

    # Graph
    edge_index = build_edge_index_from_parents(parent_indices)

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        root_pos_ctx=root_ctx_norm
    )
    data = data.to(device)

    out = model(data)
    rot_pred = out["rot"]
    root_pos_pred = out["root_pos"]

    # Denormalize root deltas
    root_delta_norm_pred = root_pos_pred.view(1, -1).view(target_len, 3)
    root_delta_pred = root_mean + root_delta_norm_pred * root_std 

    # Reconstruct root positions
    start_pos = root_pos[gap_start - 1].to(device)
    cumulative = torch.cumsum(root_delta_pred, dim=0)
    root_pred = start_pos.unsqueeze(0) + cumulative

    rot_pred = rot_pred.view(J, target_len, 6).permute(1, 0, 2).contiguous() # [J, F, 6] -> [F, J, 6]                   

    return rot_pred.cpu(), root_pred.cpu()
