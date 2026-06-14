import torch
import numpy as np

from torch_geometric.data import Data

from .utils.bvh import build_edge_index, compute_lerp, forward_kinematics_pos, global_to_local_rot
from .utils.rotation import rot_6d_to_rot_3x3, rot_3x3_to_rot_6d, compute_slerp


@torch.no_grad()
def predict_gap(model, device, rot_6d, root_pos, parent_indices, context_len_pre, context_len_post, target_len, gap_start, offsets, inner_rots, root_pos_delta_mode, rotations_delta_mode):
    
    J = rot_6d.shape[1]
    second_start = gap_start + target_len
    end = second_start + context_len_post
    first_start = gap_start - context_len_pre

    # Rotations and offsets
    first_part_rot = rot_6d[first_start : gap_start]
    second_part_rot = rot_6d[second_start : end]
    rot_ctx = np.concatenate([first_part_rot, second_part_rot], axis=0) # [F, J, 6]

    if inner_rots == "global":
        first_part_root_pos = root_pos[first_start : gap_start]
        second_part_root_pos = root_pos[second_start : end] 
        root_pos_ctx = torch.cat([first_part_root_pos, second_part_root_pos], dim=0).to(device) 
        rot_ctx_tensor = torch.from_numpy(rot_ctx).to(device, dtype=torch.float32)
        offsets_tensor = offsets.clone().detach().to(device, dtype=torch.float32)
        parents_tensor = parent_indices.clone().detach().to(device)
        _, global_3x3_rots = forward_kinematics_pos(
            offsets=offsets_tensor,
            parent_indices=parents_tensor,
            root_pos=root_pos_ctx,
            rot_6d=rot_ctx_tensor,
            local_rots=True
        )
        rot_ctx = rot_3x3_to_rot_6d(global_3x3_rots)
        rot_ctx = rot_ctx.cpu().numpy()
        
    x_feat = torch.tensor(rot_ctx, dtype=torch.float32).permute(1, 0, 2).reshape(J, -1) # [J, F * 6]
    offsets_tensor = torch.tensor(offsets, dtype=torch.float32) if not isinstance(offsets, torch.Tensor) else offsets.clone()
    parent_tensor = torch.tensor(parent_indices) if not isinstance(parent_indices, torch.Tensor) else parent_indices
    offsets_tensor[parent_tensor == -1] = 0.0
    bone_lengths = torch.linalg.norm(offsets_tensor, dim=1, keepdim=True)
    x_feat = torch.cat([x_feat, bone_lengths], dim=1) # [J, F * 6 + 1]

    # Root positions
    first_part_root_pos = root_pos[first_start : gap_start]
    second_part_root_pos = root_pos[second_start : end] 
    first_ctx_root_pos = first_part_root_pos[0].clone()
    first_part_root_pos = first_part_root_pos - first_ctx_root_pos
    second_part_root_pos = second_part_root_pos - first_ctx_root_pos 
    root_pos_ctx = torch.cat([first_part_root_pos, second_part_root_pos], dim=0).to(device) 

    # Graph
    edge_index = build_edge_index(parent_indices)

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        root_pos_ctx=root_pos_ctx
    )
    data = data.to(device)
    out = model(data)
    
    # Reshape rotations
    rot_pred_delta = out["rot"]
    rot_pred_delta = rot_pred_delta.view(J, target_len, 6).permute(1, 0, 2).contiguous() # [J, F, 6] -> [F, J, 6]  

    # Reconstruct rotations from deltas
    slerp_start_6d = rot_6d[gap_start - 1].clone().detach().to(device, dtype=torch.float32)

    if rotations_delta_mode == "linear":
        slerp_end_6d = rot_6d[second_start].clone().detach().to(device, dtype=torch.float32)
        if inner_rots == "global":
            slerps_rots = torch.cat([slerp_start_6d.unsqueeze(0), slerp_end_6d.unsqueeze(0)], dim=0) # [2, J, 6]
            slerps_root_pos = torch.cat([root_pos[gap_start - 1].unsqueeze(0), root_pos[second_start].unsqueeze(0)], dim=0).to(device) # [2, 3]
            offsets_tensor = offsets.clone().detach().to(device, dtype=torch.float32)
            parents_tensor = parent_indices.clone().detach().to(device)
            _, global_3x3_rots = forward_kinematics_pos(
                offsets=offsets_tensor,
                parent_indices=parents_tensor,
                root_pos=slerps_root_pos,
                rot_6d=slerps_rots,
                local_rots=True
            )
            global_6d_rots = rot_3x3_to_rot_6d(global_3x3_rots)
            slerp_start_6d = global_6d_rots[0]
            slerp_end_6d = global_6d_rots[1]

        rot_slerp = compute_slerp(slerp_start_6d, slerp_end_6d, target_len)

    elif rotations_delta_mode == "last_frame":
        rot_slerp = slerp_start_6d.unsqueeze(0).expand(target_len, -1, -1) # [F, J, 6]
        rot_slerp = rot_6d_to_rot_3x3(rot_slerp)

    if rotations_delta_mode == "none":
        rot_pred = rot_6d_to_rot_3x3(rot_pred_delta)
    else:
        rot_pred_delta = rot_6d_to_rot_3x3(rot_pred_delta) 
        rot_pred = torch.matmul(rot_pred_delta, rot_slerp)

    if inner_rots == "global":
        parents_tensor = parent_indices.clone().detach().to(device)
        rot_pred = global_to_local_rot(
                global_rots=rot_pred,
                parent_indices=parents_tensor
            )  
        
    rot_pred = rot_3x3_to_rot_6d(rot_pred) 

    # Reconstruct root positions from deltas
    root_pos_delta_pred = out["root_pos"]
    root_pos_delta_pred = root_pos_delta_pred.view(1, -1).view(target_len, 3) # [F, 3]
    lerp_start_pos = root_pos[gap_start - 1]

    if root_pos_delta_mode == "linear":
        lerp_end_pos = root_pos[second_start]
        root_pos_lerp = compute_lerp(lerp_start_pos, lerp_end_pos, target_len)
    elif root_pos_delta_mode == "last_frame":
        root_pos_lerp = lerp_start_pos.unsqueeze(0).expand(target_len, -1).to(device) # [F, 3]

    if root_pos_delta_mode == "none":
        root_pos_pred = root_pos_delta_pred
    else:
        root_pos_pred = root_pos_delta_pred + root_pos_lerp.to(device)        

    return rot_pred.cpu(), root_pos_pred.cpu()
