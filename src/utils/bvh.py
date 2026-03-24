import numpy as np
import torch

from bvh import Bvh 

from .rotation import euler_zyx_to_rot_6d, rot_6d_to_rot_3x3


def parse_bvh_file(filepath):
    # Returns: root_pos, rot_6d, joint_names, parent_indices, offsets (all torch.Tensor)
   
    print(f"Parsing BVH file: {filepath}")
    with open(filepath, 'r') as f:
        mocap = Bvh(f.read())

    joint_list = mocap.get_joints()
    joint_names = [j.name for j in joint_list]

    # Parent indices
    parent_indices = []
    node_to_idx = {node : i for i, node in enumerate(joint_list)}
    for node in joint_list:
        parent = node.parent
        if parent is None or parent not in node_to_idx: parent_indices.append(-1)  
        else: parent_indices.append(node_to_idx[parent])

    parent_indices = torch.tensor(parent_indices, dtype=torch.long)
            
    # Root positions
    num_frames = mocap.nframes
    root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    for f in range(0, num_frames):
        root_pos[f, 0] = float(mocap.frame_joint_channel(f, joint_list[0].name, 'Xposition'))
        root_pos[f, 1] = float(mocap.frame_joint_channel(f, joint_list[0].name, 'Yposition'))
        root_pos[f, 2] = float(mocap.frame_joint_channel(f, joint_list[0].name, 'Zposition'))

    root_pos = torch.tensor(root_pos, dtype=torch.float32)  

    # Rotations
    angles_euler = np.zeros((num_frames, len(joint_list), 3), dtype=np.float32)
    frames = np.array(mocap.frames, dtype=np.float32)

    for j, node in enumerate(joint_list):  
        ch_count = len(mocap.joint_channels(node.name))      
        node_index_end = mocap.get_joint_channels_index(node.name) + ch_count 
        node_index_start = node_index_end - 3     
        rot_indexes = list(range(node_index_start, node_index_end))
        eulers = frames[:, rot_indexes]
        angles_euler[:, j, :] = eulers

    rot_6d = euler_zyx_to_rot_6d(angles_euler, degrees=True)
    rot_6d = torch.tensor(rot_6d, dtype=torch.float32)

    # Offsets
    offsets = np.zeros((len(joint_list), 3), dtype=np.float32)
    for j, node in enumerate(joint_list):
        offset_vals = mocap.joint_offset(node.name)
        offsets[j, 0] = float(offset_vals[0])
        offsets[j, 1] = float(offset_vals[1])
        offsets[j, 2] = float(offset_vals[2])

    offsets = torch.tensor(offsets, dtype=torch.float32)

    print(f"Parsed {num_frames} frames")

    return root_pos, rot_6d, joint_names, parent_indices, offsets


def build_edge_index_from_parents(parent_indices):
    # For torch geometric edge index format 
    
    edges = []
    for child_idx, parent_idx in enumerate(parent_indices):
        if parent_idx != -1:
            edges.append([parent_idx, child_idx])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 
    
    return edge_index


def forward_kinematics_positions_batch(offsets, parent_indices, root_pos, rot_6d):
    # Offsets [J, 3], parent_indices [J], root_pos [B, F, 3], rot_6d [B, F, J, 6] -> positions [B, F, J, 3]
    
    B, F, J, _ = rot_6d.shape
    device = rot_6d.device
    rot_mats = rot_6d_to_rot_3x3(rot_6d)  

    positions = torch.zeros((B, F, J, 3), device=device)
    global_rots = torch.zeros((B, F, J, 3, 3), device=device)

    root_idx = 0
    for j in range(0, J):
        if parent_indices[j] == -1:
            root_idx = j
            break

    positions[:, :, root_idx, :] = root_pos[:, :, :]
    global_rots[:, :, root_idx, :, :] = rot_mats[:, :, root_idx, :, :]

    for j in range(0, J):
        if j == root_idx: continue   
        p = parent_indices[j]
        if p >= j: raise ValueError("Wrong parent index order for FK computation")
        parent_rot = global_rots[:, :, p, :, :].clone() 
        global_rots[:, :, j, :, :] = torch.matmul(parent_rot, rot_mats[:, :, j, :, :])
        positions[:, :, j, :] = positions[:, :, p, :] + torch.matmul(parent_rot, offsets[j])
        
    return positions


def forward_kinematics_positions(offsets, parent_indices, root_pos, rot_6d):
    # Offsets [J, 3], parent_indices [J], root_pos [F, 3], rot_6d [F, J, 6] -> positions [F, J, 3]
    
    root_pos_batched = root_pos.unsqueeze(0)      
    rot_6d_batched = rot_6d.unsqueeze(0) 

    positions_batched = forward_kinematics_positions_batch(
        offsets=offsets,
        parent_indices=parent_indices,
        root_pos=root_pos_batched,
        rot_6d=rot_6d_batched,
    )

    positions = positions_batched.squeeze(0)

    return positions


def get_joint_indices_by_name(all_joint_names, target_joint_names):
    # Returns indices of target_joint_names (torch.Tensor)

    idxs = []
    for j in target_joint_names:
        if j not in all_joint_names: raise ValueError(f"Foot joint name '{j}' not found in all joint names")
        idxs.append(all_joint_names.index(j))

    idxs = torch.tensor(idxs, dtype=torch.long)

    return idxs


def compute_foot_contact(fk_pos, foot_joint_indices, contact_height_eps, contact_velocity_eps):
    # Fk_pos: [B, F, J, 3], foot_joint_indices -> contact: [B, F, n_feet] {0, 1} (torch.Tensor)

    feet_pos = fk_pos[:, :, foot_joint_indices, :]
    h = feet_pos[:, :, :, 1] # [B, F, n_feet], Y is height          

    ground = torch.quantile(h, 0.05, dim=1, keepdim=True) # 5th percentile per foot     

    dp = feet_pos[:, 1 :, : ] - feet_pos[:, : -1, :] # [B, F - 1, n_feet, 3]                     
    dp_plane = dp[..., (0, 2)] # X, Z
    v = torch.norm(dp_plane, dim=-1)                
    v = torch.cat([torch.zeros((v.shape[0], 1, v.shape[-1]), dtype=v.dtype, device=v.device), v], dim=1) # add zero velocity for first frame  

    contact = (h <= (ground + contact_height_eps)) & (v <= contact_velocity_eps)
    contact = contact.to(torch.float32)

    return contact


def foot_skating_loss(fk_pos_pred, tgt_foot_contact, foot_joint_indices, return_elements=False):
    # Fk_pos_pred: [B, F, J, 3], tgt_foot_contact: [B, F, n_feet] -> loss
   
    feet_pos = fk_pos_pred[:, :, foot_joint_indices, :]  
    dp = feet_pos[:, 1 : ] - feet_pos[:, : -1]                 
    dp_plane = dp[..., (0, 2)]
    dp_norm = torch.norm(dp_plane, dim=-1) 

    # contact in previous and current frame
    contact_prev = tgt_foot_contact[:, : -1] 
    contact_curr = tgt_foot_contact[:, 1 : ]    

    # only penalize motion when contact is true in both frames
    contact_pair = contact_prev * contact_curr  
    weighted_motion = dp_norm * contact_pair  
    num_active = contact_pair.sum()
    
    if not return_elements: 
        num_active = torch.clamp(num_active, min=1.0)
        return weighted_motion.sum() / num_active
    else: 
        return (weighted_motion.sum(), num_active)


def get_bvh_frame_count(bvh_path):
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
        
    motion_idx = next(i for i, ln in enumerate(lines) if ln.strip().upper() == "MOTION")
    n_frames = int(lines[motion_idx + 1].split(":")[1].strip())

    return n_frames


def replace_gap_in_bvh_text(orig_text, mocap, gap_start, target_len, euler_zyx_deg, root_pred_xyz, decimals=6):
    lines = orig_text.splitlines()
    motion_idx = next(i for i, ln in enumerate(lines) if ln.strip().upper() == "MOTION")
    n_frames = int(lines[motion_idx + 1].split(":")[1].strip())
    
    frames_start_idx = motion_idx + 3
    motion_lines = lines[frames_start_idx : frames_start_idx + n_frames]
    motion_vals = [ln.strip() for ln in motion_lines]
    
    joint_list = mocap.get_joints()
    
    root_pred_xyz = root_pred_xyz.detach().cpu().numpy()
    float_format = f"{{:.{decimals}f}}"
    
    for t in range(0, target_len):
        frame_values = []
        frame_values.extend(root_pred_xyz[t, :])

        for j in range(0, len(joint_list)):
            frame_values.extend(euler_zyx_deg[t, j, :])
        
        insert_idx = gap_start + t
        motion_vals[insert_idx] = " ".join(float_format.format(v) for v in frame_values)
    
    new_lines = lines[ : frames_start_idx] + motion_vals + lines[frames_start_idx + n_frames : ]
    text = "\n".join(new_lines) + ("\n" if orig_text.endswith("\n") else "")
    
    return text
