import numpy as np
import torch

from bvh import Bvh 
from torch_geometric.utils import to_dense_batch

from .rotation import euler_to_rot_6d, rot_6d_to_rot_3x3


def parse_bvh_file(filepath):
    # -> root_pos, rot_6d, joint_names, parent_indices, offsets (all torch.Tensor), rot_order (str)
   
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

    rot_channels = mocap.joint_channels(joint_list[0].name)[-3 : ]
    rot_order = "".join([ch[0].lower() for ch in rot_channels])
    
    rot_6d = euler_to_rot_6d(angles_euler, order=rot_order, degrees=True)
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

    return root_pos, rot_6d, joint_names, parent_indices, offsets, rot_order


def build_edge_index(parent_indices):
    # -> edge_index (torch.LongTensor [2, E]) (PyG data format)
    
    edges = []
    for child_idx, parent_idx in enumerate(parent_indices):
        if parent_idx != -1:
            edges.append([parent_idx, child_idx])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 
    
    return edge_index


def forward_kinematics_pos_batch(offsets, parent_indices, root_pos, rot_6d):
    # Offsets [J, 3], parent_indices [J], root_pos [B, F, 3], rot_6d [B, F, J, 6] 
    # -> positions [B, F, J, 3], global_rots [B, F, J, 3, 3]
    
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
        
    return positions, global_rots


def forward_kinematics_pos(offsets, parent_indices, root_pos, rot_6d):
    # Offsets [J, 3], parent_indices [J], root_pos [F, 3], rot_6d [F, J, 6] 
    # -> positions [F, J, 3], global_rots [F, J, 3, 3]
    
    root_pos_batched = root_pos.unsqueeze(0)      
    rot_6d_batched = rot_6d.unsqueeze(0) 

    positions_batched, global_rots_batched = forward_kinematics_pos_batch(
        offsets=offsets,
        parent_indices=parent_indices,
        root_pos=root_pos_batched,
        rot_6d=rot_6d_batched,
    )

    positions = positions_batched.squeeze(0)
    global_3x3_rots = global_rots_batched.squeeze(0)

    return positions, global_3x3_rots


def forward_kinematics_pos_dense_batch(offsets, parent_indices, root_pos, rot_3x3, batch_index):
    # Offsets: [N_total, 3], parent_indices: [N_total], root_pos: [B, F, 3], rot_3x3: [N_total, F, 3, 3], batch_index: [N_total]
    # -> positions: [N_total, F, 3], global_rot: [N_total, F, 3, 3]

    rot_d, mask = to_dense_batch(rot_3x3, batch_index) # [B, max_J, F, 3, 3], [B, max_J] (True / False)
    offsets_d, _ = to_dense_batch(offsets, batch_index)
    parent_indices_d, _ = to_dense_batch(parent_indices, batch_index, fill_value=-1)
    
    B, max_J, F, _, _ = rot_d.shape
    device = rot_d.device
    
    pos_list = []
    rot_list = []
    
    for j in range(0, max_J):
        valid_batch = mask[:, j]
        parents = parent_indices_d[:, j]
        if (valid_batch & (parents >= j)).any(): raise ValueError("Wrong parent index order for FK computation")

        current_pos = torch.zeros((B, F, 3), device=device)
        current_rot = torch.zeros((B, F, 3, 3), device=device)
        
        is_root = valid_batch & (parents == -1)
        if is_root.any():
            current_pos[is_root, :, :] = root_pos[is_root, :, :]
            current_rot[is_root, :, :, :] = rot_d[is_root, j, :, :, :]
            
        is_child = valid_batch & (parents != -1)
        if is_child.any():
            unique_parents = parents[is_child].unique()
            for parent in unique_parents:
                parent_mask = is_child & (parents == parent)
                parent_int = parent.item()
                
                parent_pos = pos_list[parent_int][parent_mask, :, :]
                parent_rot = rot_list[parent_int][parent_mask, :, :, :]
                
                current_rot[parent_mask, :, :, :] = torch.matmul(parent_rot, rot_d[parent_mask, j, :, :, :])
                
                off_rot = torch.matmul(parent_rot, offsets_d[parent_mask, j].view(-1, 1, 3, 1)).squeeze(-1)
                current_pos[parent_mask, :, :] = parent_pos + off_rot
                
        pos_list.append(current_pos)
        rot_list.append(current_rot)
        
    pos_dense = torch.stack(pos_list, dim=1) # [B, max_J, F, 3]
    positions = pos_dense[mask] # [N_total, F, 3]

    rot_dense = torch.stack(rot_list, dim=1) # [B, max_J, F, 3, 3]
    global_3x3_rots = rot_dense[mask] # [N_total, F, 3, 3]
    
    return positions, global_3x3_rots


def get_joint_indices_by_name(all_joint_names, target_joint_names):
    # -> indices of target_joint_names (torch.Tensor)

    idxs = []
    for j in target_joint_names:
        if j not in all_joint_names: raise ValueError(f"Joint name '{j}' not found in all joint names")
        idxs.append(all_joint_names.index(j))

    idxs = torch.tensor(idxs, dtype=torch.long)

    return idxs


def compute_lerp_batch(start, end, count_to_generate):
    # Start: [B, X], End: [B, X], count_to_generate: int 
    # -> Lerp: [B, count_to_generate, X]

    t = torch.linspace(0, 1, steps=count_to_generate + 2, device=start.device)[1 : -1]
    t = t.view(1, count_to_generate, 1)
    start = start.unsqueeze(1)
    end = end.unsqueeze(1)
    result = start + t * (end - start)

    return result


def compute_lerp(start, end, count_to_generate):
    # Start: [X], End: [X], count_to_generate: int 
    # -> Lerp: [count_to_generate, X]

    t = torch.linspace(0, 1, steps=count_to_generate + 2, device=start.device)[1 : -1]
    t = t.view(count_to_generate, 1)
    start = start.unsqueeze(0)
    end = end.unsqueeze(0)
    result = start + t * (end - start)

    return result


def print_skeleton_hierarchy(joint_names, parent_indices):
    children_map = {i : [] for i in range(0, len(joint_names))}
    root_idx = 0
    
    for i, p in enumerate(parent_indices):
        p_val = p.item() if hasattr(p, 'item') else int(p)
        if p_val == -1: root_idx = i
        else: children_map[p_val].append(i)
            
    def print_node(node_idx, prefix="", is_last=True):
        branch = "└─" if is_last else "├─"
        print(f"{prefix}{branch}{node_idx} {joint_names[node_idx]}")
        new_prefix = prefix + ("  " if is_last else "│ ")
        
        children = children_map[node_idx]
        for i, child_idx in enumerate(children):
            is_last_child = (i == len(children) - 1)
            print_node(child_idx, new_prefix, is_last_child)

    print("Skeleton hierarchy")
    print(f"{root_idx} {joint_names[root_idx]}")
    
    children = children_map[root_idx]
    for i, child_idx in enumerate(children):
        is_last_child = (i == len(children) - 1)
        print_node(child_idx, "", is_last_child)
  

def compute_foot_contact(fk_pos, foot_joint_indices, contact_height_eps, contact_velocity_eps):
    # Fk_pos: [B, F, J, 3], foot_joint_indices 
    # -> contact: [B, F, n_feet] {0, 1} (torch.Tensor)

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
    # Fk_pos_pred: [B, F, J, 3], tgt_foot_contact: [B, F, n_feet] 
    # -> loss
   
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


def replace_gap_in_bvh_text(orig_text, mocap, gap_start, target_len, euler_deg, root_pred, decimals=6):
    lines = orig_text.splitlines()
    motion_idx = next(i for i, ln in enumerate(lines) if ln.strip().upper() == "MOTION")
    n_frames = int(lines[motion_idx + 1].split(":")[1].strip())
    
    frames_start_idx = motion_idx + 3
    motion_lines = lines[frames_start_idx : frames_start_idx + n_frames]
    motion_vals = [ln.strip() for ln in motion_lines]
    
    joint_list = mocap.get_joints()
    
    root_pred = root_pred.detach().cpu().numpy()
    float_format = f"{{:.{decimals}f}}"
    
    for t in range(0, target_len):
        frame_values = []
        frame_values.extend(root_pred[t, :])

        for j in range(0, len(joint_list)):
            frame_values.extend(euler_deg[t, j, :])
        
        insert_idx = gap_start + t
        motion_vals[insert_idx] = " ".join(float_format.format(v) for v in frame_values)
    
    new_lines = lines[ : frames_start_idx] + motion_vals + lines[frames_start_idx + n_frames : ]
    text = "\n".join(new_lines) + ("\n" if orig_text.endswith("\n") else "")
    
    return text


def compress_skeleton_hierarchy(parent_indices, joint_names):
    hierarchy = []
    for i, parent_idx in enumerate(parent_indices):
        current_name = joint_names[i]
        parent_name = None if parent_idx == -1 else joint_names[parent_idx]
        hierarchy.append((current_name, parent_name))

    hierarchy = sorted(hierarchy)
        
    return hierarchy
