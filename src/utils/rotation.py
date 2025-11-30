import torch

from scipy.spatial.transform import Rotation as R


def euler_zyx_to_rot_6d(euler_angles):
    # Euler angles [F, J, 3] (ZYX, rad) -> 6D [F, J, 6]
    F, J, _ = euler_angles.shape
    euler_flat = euler_angles.reshape(-1, 3) # [F * J, 3]
    r = R.from_euler('zyx', euler_flat, degrees=False)
    rot_mats = r.as_matrix() # [F * J, 3, 3]
    rot_6d = rot_mats[:, :, :2].reshape(F, J, 6)
    
    return rot_6d


def rot_6d_to_rot_3x3(rot_6d):
    # 6D [(shape), 6] -> 3 x 3 matrix [(shape), 3, 3]
    orig_shape = rot_6d.shape[ : -1]
    rot_6d_flat = rot_6d.view(-1, 6)

    a_1 = torch.stack([rot_6d_flat[:, 0], rot_6d_flat[:, 2], rot_6d_flat[:, 4]], dim=-1)
    a_2 = torch.stack([rot_6d_flat[:, 1], rot_6d_flat[:, 3], rot_6d_flat[:, 5]], dim=-1)  

    b_1 = torch.nn.functional.normalize(a_1, dim=-1)
    a_2_proj = (b_1 * a_2).sum(dim=-1, keepdim=True) * b_1
    b_2 = torch.nn.functional.normalize(a_2 - a_2_proj, dim=-1)
    b_3 = torch.cross(b_1, b_2, dim=-1)

    R_m = torch.stack([b_1, b_2, b_3], dim=-1)  
    R_m = R_m.view(*orig_shape, 3, 3)

    return R_m


def rot_6d_to_euler_zyx(rot_6d):
    # 6D [F, J, 6] -> Euler angles [F, J, 3] (ZYX, rad, numpy on CPU)
    F, J, _ = rot_6d.shape
    rot_matrix = rot_6d_to_rot_3x3(rot_6d) # [F, J, 3, 3]
    rot_matrix = rot_matrix.view(-1, 3, 3)  # [F * J, 3, 3]

    rot_matrix = rot_matrix.detach().cpu().numpy()

    euler = R.from_matrix(rot_matrix).as_euler('zyx', degrees=False)  
    euler = euler.reshape(F, J, 3)

    return euler


def geodesic_rotation_loss(pred_rot_6d, target_rot_6d):
    # Geodesic loss with mean reduction 
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
    loss = theta.mean()

    return loss