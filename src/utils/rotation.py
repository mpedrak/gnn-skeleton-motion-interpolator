import torch

from scipy.spatial.transform import Rotation as R


def euler_to_rot_6d(euler_angles):
    # Euler angles [F, J, 3] (ZYX, rad) -> 6D [F, J, 6]
    F, J, _ = euler_angles.shape
    euler_flat = euler_angles.reshape(-1, 3) # [F * J, 3]
    r = R.from_euler('zyx', euler_flat, degrees=False)
    rot_mats = r.as_matrix() # [F * J, 3, 3]
    rot_6d = rot_mats[:, :, :2].reshape(F, J, 6)
    
    return rot_6d


def rot_6d_to_euler_zyx(rot_6d):
    # 6D [F, J, 6] -> Euler angles [F, J, 3] (ZYX, rad)
    F, J, _ = rot_6d.shape
    rot_6d_flat = rot_6d.view(F * J, 6)

    a_1 = torch.stack([rot_6d_flat[:, 0], rot_6d_flat[:, 2], rot_6d_flat[:, 4]], dim=-1) # [F * J, 3]
    a_2 = torch.stack([rot_6d_flat[:, 1], rot_6d_flat[:, 3], rot_6d_flat[:, 5]], dim=-1)  

    b_1 = torch.nn.functional.normalize(a_1, dim=-1)
    a_2_proj = (b_1 * a_2).sum(dim=-1, keepdim=True) * b_1
    b_2 = torch.nn.functional.normalize(a_2 - a_2_proj, dim=-1)
    b_3 = torch.cross(b_1, b_2, dim=-1)

    Rm = torch.stack([b_1, b_2, b_3], dim=-1)  
    Rm_np = Rm.detach().cpu().numpy()
    euler = R.from_matrix(Rm_np).as_euler('zyx', degrees=False)  
    euler = euler.reshape(F, J, 3)

    return euler
