import numpy as np
import torch

from bvh import Bvh 

from .rotation import euler_zyx_to_rot_6d


def parse_bvh_file(filepath):
    print(f"Parsing BVH file: {filepath}")
    
    with open(filepath, 'r') as f:
        mocap = Bvh(f.read())

    joint_list = mocap.get_joints()
    joint_names = [j.name for j in joint_list]
    parent_indices = []
    node_to_idx = {node : i for i, node in enumerate(joint_list)}
    for node in joint_list:
        parent = node.parent
        if parent is None or parent not in node_to_idx: 
            parent_indices.append(-1)  
        else: 
            parent_indices.append(node_to_idx[parent])
            
    num_frames = mocap.nframes
    root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    for f in range(0, num_frames):
        root_pos[f, 0] = float(mocap.frame_joint_channel(f, joint_list[0].name, 'Xposition'))
        root_pos[f, 1] = float(mocap.frame_joint_channel(f, joint_list[0].name, 'Yposition'))
        root_pos[f, 2] = float(mocap.frame_joint_channel(f, joint_list[0].name, 'Zposition'))

    rot_channels = ['Zrotation', 'Yrotation', 'Xrotation']
    angles_euler = np.zeros((num_frames, len(joint_list), 3), dtype=np.float32) # [F, J, 3]

    for j, node in enumerate(joint_list):
        for i, ch in enumerate(rot_channels):
            angles_euler[:, j, i] = np.array([
                float(mocap.frame_joint_channel(f, node.name, ch)) * np.pi / 180.0
                for f in range(0, num_frames)], dtype=np.float32)

    rot_6d = euler_zyx_to_rot_6d(angles_euler) # [F, J, 6]
    rot_6d = torch.tensor(rot_6d, dtype=torch.float32)

    print(f"Parsed {num_frames} frames")

    return root_pos, rot_6d, joint_names, parent_indices


def build_edge_index_from_parents(parent_indices):
    edges = []
    for child_idx, parent_idx in enumerate(parent_indices):
        if parent_idx != -1:
            edges.append([parent_idx, child_idx])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() # For torch geometric edge_index format
    return edge_index


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


def compute_root_deltas(root_pos):
    # [F, 3] (numpy) -> [F, 3] (deltas, torch.FloatTensor)
    deltas = np.zeros_like(root_pos, dtype=np.float32)
    deltas[1 : ] = root_pos[1 : ] - root_pos[ : -1]
    deltas = torch.tensor(deltas, dtype=torch.float32)
    return deltas
