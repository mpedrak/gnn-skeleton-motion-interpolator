import torch
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R


def euler_to_rot_6d(euler_angles, order, degrees):
    # Euler angles [(shape), 3] (order) (degrees or radians)
    # -> 6D [(shape), 6]
    
    orig_shape = euler_angles.shape[ : -1]
    euler_flat = euler_angles.reshape(-1, 3) 

    reversed_order = order[ :: -1]
    euler_flat = euler_flat[:, [2, 1, 0]]
    r = R.from_euler(seq=reversed_order, angles=euler_flat, degrees=degrees)
    
    rot_mats = r.as_matrix() # [F * J, 3, 3]
    rot_6d = rot_mats[:, :, :2].reshape(*orig_shape, 6)
    
    return rot_6d


def rot_6d_to_rot_3x3(rot_6d):
    # 6D [(shape), 6] 
    # -> 3 x 3 matrix [(shape), 3, 3]
    
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


def rot_6d_to_euler(rot_6d, order, degrees):
    # 6D [(shape), 6] 
    # -> Euler angles [(shape), 3] (order) (degrees or radians) (numpy on CPU)
    
    rot_matrix = rot_6d_to_rot_3x3(rot_6d) # [(shape), 3, 3]
    orig_shape = rot_matrix.shape[ : -2] 
    rot_matrix = rot_matrix.view(-1, 3, 3)  

    rot_matrix = rot_matrix.detach().cpu().numpy()

    reversed_order = order[ :: -1]
    euler = R.from_matrix(rot_matrix).as_euler(seq=reversed_order, degrees=degrees)  
    euler = euler[:, [2, 1, 0]]
    euler = euler.reshape(*orig_shape, 3)

    return euler


def rot_6d_to_quat_numpy(rot_6d):
    # 6D [(shape), 6] 
    # -> Quaternions [(shape), 4] (x, y, z, w)
    
    orig_shape = rot_6d.shape[ : -1]
    device = rot_6d.device
    
    rot_matrix = rot_6d_to_rot_3x3(rot_6d) 
    rot_matrix_flat = rot_matrix.view(-1, 3, 3)
    
    rot_matrix_np = rot_matrix_flat.detach().cpu().numpy()
    
    quat_np = R.from_matrix(rot_matrix_np).as_quat()
    
    quat = torch.tensor(quat_np, dtype=rot_6d.dtype, device=device)
    quat = quat.view(*orig_shape, 4)
    
    return quat


def rot_6d_to_quat_torch(rot_6d):
    # 6D [(shape), 6] 
    # -> Quaternions [(shape), 4] (x, y, z, w)
      
    orig_shape = rot_6d.shape[ : -1]
    
    rot_matrix = rot_6d_to_rot_3x3(rot_6d) 
    rot_matrix_flat = rot_matrix.view(-1, 3, 3)

    m00 = rot_matrix_flat[:, 0, 0]
    m01 = rot_matrix_flat[:, 0, 1]
    m02 = rot_matrix_flat[:, 0, 2]
    
    m10 = rot_matrix_flat[:, 1, 0]
    m11 = rot_matrix_flat[:, 1, 1]
    m12 = rot_matrix_flat[:, 1, 2]
    
    m20 = rot_matrix_flat[:, 2, 0]
    m21 = rot_matrix_flat[:, 2, 1]
    m22 = rot_matrix_flat[:, 2, 2]

    q_abs = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1)
    q_abs = torch.sqrt(torch.clamp(q_abs, min=0.0))

    quat_by_w = torch.stack([q_abs[:, 0], m21 - m12, m02 - m20, m10 - m01], dim=-1)
    quat_by_x = torch.stack([m21 - m12, q_abs[:, 1], m01 + m10, m02 + m20], dim=-1)
    quat_by_y = torch.stack([m02 - m20, m01 + m10, q_abs[:, 2], m12 + m21], dim=-1)
    quat_by_z = torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[:, 3]], dim=-1)

    idx = torch.argmax(q_abs, dim=-1)
    
    quat = torch.empty_like(quat_by_w)
    quat[idx == 0] = quat_by_w[idx == 0]
    quat[idx == 1] = quat_by_x[idx == 1]
    quat[idx == 2] = quat_by_y[idx == 2]
    quat[idx == 3] = quat_by_z[idx == 3]

    quat = torch.nn.functional.normalize(quat, dim=-1)

    quat = quat[:, [1, 2, 3, 0]]
    
    quat = quat.view(*orig_shape, 4)
    
    return quat


def quat_to_rot_3x3(quat):
    # Quaternions [(shape), 4] (x, y, z, w) 
    # -> 3 x 3 matrix [(shape), 3, 3]
    
    orig_shape = quat.shape[ : -1]
    quat_flat = quat.reshape(-1, 4)

    x = quat_flat[:, 0]
    y = quat_flat[:, 1]
    z = quat_flat[:, 2]
    w = quat_flat[:, 3]

    x2 = x + x
    y2 = y + y
    z2 = z + z

    xx = x * x2
    yy = y * y2
    zz = z * z2
    xy = x * y2
    yz = y * z2
    xz = x * z2
    wx = w * x2
    wy = w * y2
    wz = w * z2

    r00 = 1.0 - (yy + zz)
    r01 = xy - wz
    r02 = xz + wy

    r10 = xy + wz
    r11 = 1.0 - (xx + zz)
    r12 = yz - wx

    r20 = xz - wy
    r21 = yz + wx
    r22 = 1.0 - (xx + yy)

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)

    R_m = torch.stack([row0, row1, row2], dim=-2)  
    R_m = R_m.view(*orig_shape, 3, 3)

    return R_m


def compute_slerp_batch(start_6d, end_6d, count_to_generate):
    # Start: [N_total, 6], End: [N_total, 6], count_to_generate: int 
    # -> Slerp (3 x 3 matrix): [N_total, count_to_generate, 3, 3]

    q0 = rot_6d_to_quat_torch(start_6d) 
    q1 = rot_6d_to_quat_torch(end_6d) 

    t = torch.linspace(0, 1, steps=count_to_generate + 2, device=start_6d.device)[1 : -1]
    t = t.view(1, count_to_generate, 1)
    
    q0 = q0.unsqueeze(1)
    q1 = q1.unsqueeze(1)

    cos_half_theta = (q0 * q1).sum(dim=-1, keepdim=True)
    flip_mask = cos_half_theta < 0
    q1 = torch.where(flip_mask, -q1, q1)
    cos_half_theta = torch.abs(cos_half_theta)

    mask = cos_half_theta > 0.9995
    half_theta = torch.acos(torch.clamp(cos_half_theta, -1.0, 1.0))
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta**2) + 1e-8 

    ratio_a = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = torch.sin(t * half_theta) / sin_half_theta

    res_slerp = ratio_a * q0 + ratio_b * q1
    res_lerp = q0 + t * (q1 - q0)
    res_lerp = F.normalize(res_lerp, dim=-1)

    q_interpolated = torch.where(mask, res_lerp, res_slerp)

    rot_3x3 = quat_to_rot_3x3(q_interpolated) 

    return rot_3x3


def compute_slerp(start_6d, end_6d, count_to_generate):
    # Start: [J, 6] End: [J, 6], count_to_generate: int 
    # -> Slerp (3 x 3 matrix): [count_to_generate, J, 3, 3]
   
    result = compute_slerp_batch(start_6d, end_6d, count_to_generate)
    result = result.transpose(0, 1)

    return result


def rot_3x3_to_rot_6d(rot_matrix):
    # 3 x 3 matrix [(shape), 3, 3] 
    # -> 6D [(shape), 6]
    
    orig_shape = rot_matrix.shape[ : -2]
    rot_6d = rot_matrix[..., :, : 2].reshape(*orig_shape, 6)
    
    return rot_6d