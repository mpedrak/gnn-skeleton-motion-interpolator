import torch

from .rotation import rot_6d_to_rot_3x3, rot_6d_to_quat


def geodesic_rotation_loss(pred_rot_6d, target_rot_6d, reduction='mean'):
    
    JxB, Fx6 = pred_rot_6d.shape
    F = Fx6 // 6
    pred_rot_6d = pred_rot_6d.view(JxB, F, 6)
    target_rot_6d = target_rot_6d.view(JxB, F, 6)

    R_pred = rot_6d_to_rot_3x3(pred_rot_6d)   
    R_target = rot_6d_to_rot_3x3(target_rot_6d)  

    R_rel = torch.matmul(R_pred.transpose(-1, -2), R_target) 
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]

    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

    theta = torch.acos(cos_theta)  

    if reduction == 'mean': return theta.mean()
    elif reduction == 'sum': return theta.sum()
    elif reduction == 'none': return theta
    else: raise ValueError(f"Unsupported reduction: {reduction}")


def calculate_l2p(pred_pos, target_pos, reduction='mean'):

    distances = torch.linalg.norm(pred_pos - target_pos, dim=-1)
  
    if reduction == 'mean': return distances.mean()
    elif reduction == 'sum': return distances.sum()
    elif reduction == 'none': return distances
    else: raise ValueError(f"Unsupported reduction: {reduction}")


def calculate_l2q(pred_rot_6d, target_rot_6d, reduction='mean'):

    pred_q = rot_6d_to_quat(pred_rot_6d.view(-1, 6))
    target_q = rot_6d_to_quat(target_rot_6d.view(-1, 6))

    diff_1 = torch.linalg.norm(pred_q - target_q, dim=-1)
    diff_2 = torch.linalg.norm(pred_q + target_q, dim=-1)

    min_diff = torch.minimum(diff_1, diff_2)

    if reduction == 'mean': return min_diff.mean()
    elif reduction == 'sum': return min_diff.sum()
    elif reduction == 'none': return min_diff
    else: raise ValueError(f"Unsupported reduction: {reduction}")

def calculate_npss(pred, target, reduction='mean', eps=1e-8):
 
    fft_pred = torch.fft.rfft(pred, dim=-1) 
    fft_target = torch.fft.rfft(target, dim=-1)
     
    power_pred = torch.abs(fft_pred) ** 2
    power_target = torch.abs(fft_target) ** 2
    
    sum_power_pred = power_pred.sum(dim=-1, keepdim=True) + eps
    sum_power_target = power_target.sum(dim=-1, keepdim=True) + eps
    
    norm_power_pred = power_pred / sum_power_pred
    norm_power_target = power_target / sum_power_target
    
    cdf_pred = torch.cumsum(norm_power_pred, dim=-1)
    cdf_target = torch.cumsum(norm_power_target, dim=-1)
    
    emd = torch.sum(torch.abs(cdf_pred - cdf_target), dim=-1)  
    
    if reduction == 'mean': return emd.mean()
    elif reduction == 'sum': return emd.sum()
    elif reduction == 'none': return emd
    else: raise ValueError(f"Unsupported reduction: {reduction}")


def compute_smoothness_loss(fk_pos_pred, fk_pos_tgt, order, reduction='mean'):
    # Compute smoothness loss of specified order (velocity = 1, acceleration = 2, jerk = 3)

    if order < 1: raise ValueError("Order must be >= 1")
    v_pred = fk_pos_pred.clone()
    v_tgt = fk_pos_tgt.clone()
    for _ in range(order):
        v_pred = v_pred[:, 1:, ...] - v_pred[:, :-1, ...]
        v_tgt = v_tgt[:, 1:, ...] - v_tgt[:, :-1, ...]

    v = v_pred - v_tgt

    if reduction == 'mean': return torch.abs(v).mean()
    elif reduction == 'sum': return torch.abs(v).sum()
    elif reduction == 'none': return torch.abs(v)
    else: raise ValueError(f"Unsupported reduction: {reduction}")


def foot_contact_loss(foot_contact_tgt, foot_contact_pred, reduction='mean'):

    diff = foot_contact_pred - foot_contact_tgt
    
    if reduction == 'mean': return torch.abs(diff).mean()
    elif reduction == 'sum': return torch.abs(diff).sum()
    elif reduction == 'none': return torch.abs(diff)
    else: raise ValueError(f"Unsupported reduction: {reduction}")
