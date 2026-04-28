import torch

from .utils.metrics import calculate_smoothness_loss, geodesic_rotation_loss
from .utils.bvh import forward_kinematics_pos_dense_batch, compute_lerp_batch
from .utils.rotation import rot_6d_to_rot_3x3, compute_slerp_batch


def calculate_loss(out, batch, l1_func, l2_func, loss_weights):
    
    # Rotations
    rot_pred_delta = out["rot"]
    N_total, Fx6 = rot_pred_delta.shape
    F_target = Fx6 // 6 
    rot_pred_delta = rot_pred_delta.view(N_total, F_target, 6)

    rot_6d_for_slerp = batch.rot_6d_for_slerp # [N_total, 2, 6]
    slerp_start_6d = rot_6d_for_slerp[:, 0, :]
    slerp_end_6d = rot_6d_for_slerp[:, 1, :]
    rot_slerp = compute_slerp_batch(slerp_start_6d, slerp_end_6d, F_target) 
    
    rot_pred_delta_3x3 = rot_6d_to_rot_3x3(rot_pred_delta)
    rot_pred_3x3 = torch.matmul(rot_pred_delta_3x3, rot_slerp) # [N_total, F_target, 3, 3]

    rot_tgt_3x3 = rot_6d_to_rot_3x3(batch.y.view(N_total, F_target, 6))
    rot_geo_loss = geodesic_rotation_loss(pred_rot_3x3=rot_pred_3x3, target_rot_3x3=rot_tgt_3x3)

    # Root positions
    root_pos_delta_pred = out['root_pos']
    root_pos_delta_pred = root_pos_delta_pred.view(batch.num_graphs, F_target, 3)
    root_pos_for_lerp = batch.root_pos_for_lerp.view(batch.num_graphs, 2, 3)
    lerp_start_pos = root_pos_for_lerp[:, 0, :] 
    lerp_end_pos = root_pos_for_lerp[:, 1, :]
    root_pos_lerp = compute_lerp_batch(lerp_start_pos, lerp_end_pos, F_target)
    root_pos_pred = root_pos_delta_pred + root_pos_lerp
    
    root_pos_pred_flat = root_pos_pred.view(batch.num_graphs, -1) # [B, F_target * 3]
    root_pos_tgt_flat = batch.root_pos_tgt.view(batch.num_graphs, -1)
    root_pos_loss = l2_func(root_pos_pred_flat, root_pos_tgt_flat)

    # Forward kinematics 
    fk_pos_pred, _ = forward_kinematics_pos_dense_batch(
        offsets=batch.offsets,
        parent_indices=batch.parent_indices,
        root_pos=root_pos_pred,
        rot_3x3=rot_pred_3x3,
        batch_index=batch.batch
    )

    fk_pos_tgt_flat = batch.fk_pos_tgt.view(N_total, -1)
    fk_pos_pred_flat = fk_pos_pred.view(N_total, -1)
    fk_pos_loss = l1_func(fk_pos_pred_flat, fk_pos_tgt_flat)

    fk_pos_tgt = batch.fk_pos_tgt.view(N_total, F_target, 3)
    sm_1_loss = calculate_smoothness_loss(fk_pos_pred, fk_pos_tgt, order=1, reduction='mean')
    sm_2_loss = calculate_smoothness_loss(fk_pos_pred, fk_pos_tgt, order=2, reduction='mean')
    sm_3_loss = calculate_smoothness_loss(fk_pos_pred, fk_pos_tgt, order=3, reduction='mean')

    # Total loss
    rot_geo_loss_w = loss_weights['rot_geo_l1'] * rot_geo_loss
    root_pos_loss_w = loss_weights['root_pos_l2'] * root_pos_loss
    fk_pos_loss_w = loss_weights['fk_pos_l1'] * fk_pos_loss
    sm_vel_loss_w = loss_weights['sm_vel'] * sm_1_loss
    sm_acc_loss_w = loss_weights['sm_acc'] * sm_2_loss
    sm_jerk_loss_w = loss_weights['sm_jerk'] * sm_3_loss

    loss = rot_geo_loss_w + root_pos_loss_w + fk_pos_loss_w + sm_vel_loss_w + sm_acc_loss_w + sm_jerk_loss_w

    return loss, rot_geo_loss_w, root_pos_loss_w, fk_pos_loss_w, sm_vel_loss_w, sm_acc_loss_w, sm_jerk_loss_w
