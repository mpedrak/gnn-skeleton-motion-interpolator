import torch

from .utils.metrics import geodesic_rotation_loss
from .utils.bvh import forward_kinematics_positions_batch


def calculate_loss(out, batch, l1_func, l2_func, root_std, root_mean, offsets, parent_indices, loss_weights):
    
    # Rotations
    rot_pred = out["rot"]
    rot_geo_loss = geodesic_rotation_loss(pred_rot_6d=rot_pred, target_rot_6d=batch.y)

    # Root positions
    BxJ, Fx6 = rot_pred.shape
    F_target = Fx6 // 6
    J = BxJ // batch.num_graphs    
    rot_pred = rot_pred.view(batch.num_graphs, J, F_target, 6).permute(0, 2, 1, 3) # [B, F_target, J, 6]
    
    root_pos_pred = out['root_pos']
    root_pos_pred = root_pos_pred.view(batch.num_graphs, F_target, 3)
    root_pos_pred = root_pos_pred * root_std + root_mean

    last_root_pos_absolute = batch.last_root_pos_absolute.view(batch.num_graphs, 1, 3) # position at target start frame - 1
    root_pos_pred_absolute = last_root_pos_absolute + torch.cumsum(root_pos_pred, dim=1) # [B, F_target, 3]
    
    root_pos_pred = root_pos_pred_absolute.view(batch.num_graphs, -1) # [B, F_target * 3]
    root_pos_tgt = batch.root_pos_tgt.view(batch.num_graphs, -1)
    root_pos_loss = l2_func(root_pos_pred, root_pos_tgt)

    # Forward kinematics 
    fk_pos_pred = forward_kinematics_positions_batch(
        offsets=offsets,
        parent_indices=parent_indices,
        root_pos=root_pos_pred_absolute,
        rot_6d=rot_pred
    ) 

    fk_pos_tgt = batch.fk_pos.view(batch.num_graphs, -1)
    fk_pos_pred = fk_pos_pred.view(batch.num_graphs, -1)
    fk_pos_loss = l1_func(fk_pos_pred, fk_pos_tgt)

    # Total loss
    rot_geo_loss_w = loss_weights['rot_geo_l1'] * rot_geo_loss
    root_pos_loss_w = loss_weights['root_pos_l2'] * root_pos_loss
    fk_pos_loss_w = loss_weights['fk_pos_l1'] * fk_pos_loss
    loss = rot_geo_loss_w + root_pos_loss_w + fk_pos_loss_w

    return loss, rot_geo_loss_w, root_pos_loss_w, fk_pos_loss_w
