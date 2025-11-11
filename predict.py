import os
import sys
import yaml
import torch
import numpy as np

from bvh import Bvh
from torch_geometric.data import Data

from src.model import SkeletalMotionInterpolator
from src.utils.bvh import build_edge_index_from_parents, extract_prev_euler_from_bvh, replace_gap_in_bvh_text, parse_bvh_file
from src.utils.rotation import rot_6d_to_euler_zyx, unwrap_euler_sequence


config_path = "./config/cfg.yaml"
with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
input_bvh_path = "data/predict/test_3.bvh"
gap_start_frame = 70

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


@torch.no_grad()
def predict_gap(model, device, rot_6d, root_pos, parent_indices, context_len_pre, context_len_post, 
                target_len, gap_start, root_mean, root_std):
    
    J = rot_6d.shape[1]
    second_start = gap_start + target_len
    
    first_part_rot = rot_6d[gap_start - context_len_pre : gap_start]
    second_part_rot = rot_6d[second_start : second_start + context_len_post]
    rot_ctx = np.concatenate([first_part_rot, second_part_rot], axis=0)
    x_feat = torch.tensor(rot_ctx, dtype=torch.float32).permute(1, 0, 2).reshape(J, -1) # [J, F, 6] -> [J, F * 6]

    first_part_root = root_pos[gap_start - context_len_pre : gap_start]
    second_part_root = root_pos[second_start : second_start + context_len_post]
    root_ctx_raw = np.concatenate([first_part_root, second_part_root], axis=0)

    root_ctx_norm = (torch.tensor(root_ctx_raw, dtype=torch.float32, device=device) - root_mean) / root_std
    root_ctx_norm = root_ctx_norm.reshape(-1)

    edge_index = build_edge_index_from_parents(parent_indices)

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        root_ctx_norm=root_ctx_norm
    ).to(device)

    out = model(data)
    rot_pred = out["rot"]
    root_norm_pred = out["root_norm"]

    rot_pred = rot_pred.view(J, target_len, 6).permute(1, 0, 2).contiguous() # [J, F, 6] -> [F, J, 6]
    root_norm_pred = root_norm_pred.view(1, -1).view(target_len, 3)            
    root_pred = root_mean + root_norm_pred * root_std                     

    return rot_pred.cpu(), root_pred.cpu()


model_path = config["model_path"]
root_stats_path = config["root_stats_path"]

stats = np.load(root_stats_path)
root_mean = torch.tensor(stats["mean"], dtype=torch.float32)
root_std = torch.tensor(stats["std"], dtype=torch.float32)

with open(input_bvh_path, "r") as f:
    text = f.read()

mocap = Bvh(text)
root_pos, rot_6d, joint_names, parent_indices = parse_bvh_file(input_bvh_path)
frames_total = rot_6d.shape[0]

model = SkeletalMotionInterpolator(
    context_len_pre=config["context_len_pre"],
    context_len_post=config["context_len_post"],
    target_len=config["target_len"],
    hidden_dim=config["hidden_dim"],
    hidden_layers=config["hidden_layers"],
    root_pos_hidden_dim=config["root_pos_hidden_dim"],
    heads=config["heads"],
    dropout=config["dropout"],
    node_features=config["node_features"],
    graph_features=config["graph_features"]
)

model = model.to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Loaded checkpoint: {model_path}")

context_len_pre = config["context_len_pre"]
context_len_post = config["context_len_post"]
target_len = config["target_len"]

with torch.no_grad():
    rot_pred, root_pred = predict_gap(
        model=model,
        device=device,
        rot_6d=rot_6d,
        root_pos=root_pos,
        parent_indices=parent_indices,
        context_len_pre=context_len_pre,
        context_len_post=context_len_post,
        target_len=target_len,
        gap_start=gap_start_frame,
        root_mean=root_mean.to(device),
        root_std=root_std.to(device),
    )

euler_zyx_deg_rad = rot_6d_to_euler_zyx(rot_pred)
euler_zyx_deg = np.rad2deg(euler_zyx_deg_rad)
prev_euler_deg = extract_prev_euler_from_bvh(mocap, gap_start_frame - 1)
euler_zyx_deg = unwrap_euler_sequence(euler_zyx_deg, prev_deg=prev_euler_deg)

new_text = replace_gap_in_bvh_text(
    orig_text=text,
    mocap=mocap,
    gap_start=gap_start_frame,
    target_len=target_len,
    euler_zyx_deg=euler_zyx_deg,
    root_pred_xyz=root_pred
)

out_path = os.path.splitext(input_bvh_path)[0] + "_pred.bvh"
with open(out_path, "w") as f:
    f.write(new_text)

print(f"Saved predicted BVH to: {out_path}")
print(f"Replaced frames: [{gap_start_frame + 1}, {gap_start_frame + target_len}]")
