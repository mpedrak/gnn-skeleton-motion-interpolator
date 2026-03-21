import torch

from scipy.spatial.transform import Rotation as R


def euler_zyx_to_rot_6d(euler_angles):
    # Euler angles [(shape), 3] (ZYX, rad) -> 6D [(shape), 6]
    
    orig_shape = euler_angles.shape[ : -1]
    euler_flat = euler_angles.reshape(-1, 3) 
    euler_flat = euler_flat[:, [2, 1, 0]]
    r = R.from_euler('xyz', euler_flat, degrees=False)
    rot_mats = r.as_matrix() # [F * J, 3, 3]
    rot_6d = rot_mats[:, :, :2].reshape(*orig_shape, 6)
    
    return rot_6d


def rot_6d_to_rot_3x3(rot_6d):
    # 6D [(shape), 6] -> 3 x 3 matrix [(shape), 3, 3]
    
    orig_shape = rot_6d.shape[ : -1]
    rot_6d_flat = rot_6d.reshape(-1, 6)

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
    # 6D [(shape), 6] -> Euler angles [(shape), 3] (ZYX, rad, numpy on CPU)
    
    rot_matrix = rot_6d_to_rot_3x3(rot_6d) # [(shape), 3, 3]
    orig_shape = rot_matrix.shape[ : -2] 
    rot_matrix = rot_matrix.view(-1, 3, 3)  

    rot_matrix = rot_matrix.detach().cpu().numpy()

    euler = R.from_matrix(rot_matrix).as_euler('xyz', degrees=False)  
    euler = euler[:, [2, 1, 0]]
    euler = euler.reshape(*orig_shape, 3)

    return euler


def rot_6d_to_quat(rot_6d):
    # 6D [(shape), 6] -> Quaternions [(shape), 4] (x, y, z, w)
    
    orig_shape = rot_6d.shape[ : -1]
    device = rot_6d.device
    
    rot_matrix = rot_6d_to_rot_3x3(rot_6d) 
    rot_matrix_flat = rot_matrix.view(-1, 3, 3)
    
    rot_matrix_np = rot_matrix_flat.detach().cpu().numpy()
    
    quat_np = R.from_matrix(rot_matrix_np).as_quat()
    
    quat = torch.tensor(quat_np, dtype=rot_6d.dtype, device=device)
    quat = quat.view(*orig_shape, 4)
    
    return quat