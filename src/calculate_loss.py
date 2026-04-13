import torch

from .utils.metrics import geodesic_rotation_loss
from .utils.bvh import forward_kinematics_positions_batch
from .utils.various import compute_lerp_batch, compute_slerp_batch
from .utils.rotation import rot_6d_to_rot_3x3, rot_3x3_to_rot_6d


def calculate_loss(out, batch, l1_func, l2_func, offsets, parent_indices, loss_weights):
    
    # Rotations
    rot_pred_delta = out["rot"]
    BxJ, Fx6 = rot_pred_delta.shape
    F_target = Fx6 // 6
    J = BxJ // batch.num_graphs    
    rot_pred_delta = rot_pred_delta.view(batch.num_graphs, J, F_target, 6).permute(0, 2, 1, 3) # [B, F_target, J, 6]

    rot_6d_for_slerp = batch.rot_6d_for_slerp.view(batch.num_graphs, 2, J, 6) # [B, 2, J, 6]
    slerp_start_6d = rot_6d_for_slerp[:, 0, :, :]
    slerp_end_6d = rot_6d_for_slerp[:, 1, :, :]
    rot_slerp = compute_slerp_batch(slerp_start_6d, slerp_end_6d, F_target) 
    
    rot_pred_delta = rot_6d_to_rot_3x3(rot_pred_delta)
    rot_pred = torch.matmul(rot_pred_delta, rot_slerp)
    rot_pred = rot_3x3_to_rot_6d(rot_pred) 
    
    rot_pred_for_geo = rot_pred.permute(0, 2, 1, 3).reshape(BxJ, Fx6)
    rot_geo_loss = geodesic_rotation_loss(pred_rot_6d=rot_pred_for_geo, target_rot_6d=batch.y)
    
    # Root positions
    root_pos_delta_pred = out['root_pos']
    root_pos_delta_pred = root_pos_delta_pred.view(batch.num_graphs, F_target, 3)
    root_pos_for_lerp = batch.root_pos_for_lerp.view(batch.num_graphs, 2, 3) # [B, 2, 3]
    lerp_start_pos = root_pos_for_lerp[:, 0, :] 
    lerp_end_pos = root_pos_for_lerp[:, 1, :]
    root_pos_lerp = compute_lerp_batch(lerp_start_pos, lerp_end_pos, F_target)
    root_pos_pred = root_pos_delta_pred + root_pos_lerp
    
    root_pos_pred_flat = root_pos_pred.view(batch.num_graphs, -1) # [B, F_target * 3]
    root_pos_tgt_flat = batch.root_pos_tgt.view(batch.num_graphs, -1)
    root_pos_loss = l2_func(root_pos_pred_flat, root_pos_tgt_flat)

    # Forward kinematics 
    fk_pos_pred = forward_kinematics_positions_batch(
        offsets=offsets,
        parent_indices=parent_indices,
        root_pos=root_pos_pred,
        rot_6d=rot_pred
    ) 

    fk_pos_tgt_flat = batch.fk_pos.view(batch.num_graphs, -1)
    fk_pos_pred_flat = fk_pos_pred.view(batch.num_graphs, -1)
    fk_pos_loss = l1_func(fk_pos_pred_flat, fk_pos_tgt_flat)

    # Total loss
    rot_geo_loss_w = loss_weights['rot_geo_l1'] * rot_geo_loss
    root_pos_loss_w = loss_weights['root_pos_l2'] * root_pos_loss
    fk_pos_loss_w = loss_weights['fk_pos_l1'] * fk_pos_loss
    loss = rot_geo_loss_w + root_pos_loss_w + fk_pos_loss_w

    return loss, rot_geo_loss_w, root_pos_loss_w, fk_pos_loss_w
