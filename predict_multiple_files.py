import os
import yaml
import torch
import numpy as np
import argparse

from bvh import Bvh
from torch_geometric.data import Data

from src.model import SkeletalMotionInterpolator
from src.utils.bvh import build_edge_index_from_parents, replace_gap_in_bvh_text, parse_bvh_file, compute_root_deltas, get_bvh_frame_count
from src.utils.rotation import rot_6d_to_euler_zyx


# Config
predict_data_dir = "./data/predict/"
bvh_files = ["aiming", "dance", "fight", "ground", "jumps", "run", "walk"]
hole_starts = [i for i in range(20, 480 + 1, 20)]

config_dir = "./config/"
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()

config_path = config_dir + args.config + ".yaml"
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
        
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Prediction function
@torch.no_grad()
def predict_gap(model, device, rot_6d, root_pos, parent_indices, context_len_pre, context_len_post, 
                target_len, gap_start, root_mean, root_std):
    
    J = rot_6d.shape[1]
    second_start = gap_start + target_len
    end = second_start + context_len_post
    first_start = gap_start - context_len_pre

    # Rotations
    first_part_rot = rot_6d[first_start : gap_start]
    second_part_rot = rot_6d[second_start : end]
    rot_ctx = np.concatenate([first_part_rot, second_part_rot], axis=0)
    x_feat = torch.tensor(rot_ctx, dtype=torch.float32).permute(1, 0, 2).reshape(J, -1) # [J, F * 6]

    # Root positions
    first_part_root_pos = root_pos[first_start : gap_start]
    first_part_root_pos = compute_root_deltas(first_part_root_pos)
    second_part_root_pos = root_pos[second_start : end]
    second_part_root_pos = compute_root_deltas(second_part_root_pos)
    
    root_ctx_pos = torch.cat([first_part_root_pos, second_part_root_pos], dim=0).to(device) 
    root_ctx_norm = (root_ctx_pos - root_mean) / root_std

    # Graph
    edge_index = build_edge_index_from_parents(parent_indices)

    data = Data(
        x=x_feat,
        edge_index=edge_index,
        root_pos_ctx=root_ctx_norm
    ).to(device)

    out = model(data)
    rot_pred = out["rot"]
    root_pos_pred = out["root_pos"]

    # Denormalize root deltas
    root_delta_norm_pred = root_pos_pred.view(1, -1).view(target_len, 3)
    root_delta_pred = root_mean + root_delta_norm_pred * root_std 

    # Reconstruct root positions
    start_pos = root_pos[gap_start - 1].to(device)
    cumulative = torch.cumsum(root_delta_pred, dim=0)
    root_pred = start_pos.unsqueeze(0) + cumulative

    rot_pred = rot_pred.view(J, target_len, 6).permute(1, 0, 2).contiguous() # [J, F, 6] -> [F, J, 6]                   

    return rot_pred.cpu(), root_pred.cpu()

# Model
model_path = config["model_path"]
root_stats_path = config["root_stats_path"]

stats = np.load(root_stats_path)
root_mean = torch.tensor(stats["mean"], dtype=torch.float32)
root_std = torch.tensor(stats["std"], dtype=torch.float32)

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

# Prediction
context_len_pre = config["context_len_pre"]
context_len_post = config["context_len_post"]
target_len = config["target_len"]

for bvh_file in bvh_files:
    input_bvh_path = os.path.join(predict_data_dir, bvh_file + ".bvh")
    n_frames = get_bvh_frame_count(input_bvh_path)
    with open(input_bvh_path, "r") as f:
        text = f.read()

    mocap = Bvh(text)
    root_pos, rot_6d, joint_names, parent_indices, _ = parse_bvh_file(input_bvh_path)
    frames_total = rot_6d.shape[0]   
    new_text = text  

    for hole_start in hole_starts:
        gap_start_frame = hole_start - 1 # 0 based index in code

        if gap_start_frame <= context_len_pre or gap_start_frame >= n_frames - context_len_post - target_len:
            raise ValueError("Invalid gap start frame")   

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

        new_text = replace_gap_in_bvh_text(
            orig_text=new_text,
            mocap=mocap,
            gap_start=gap_start_frame,
            target_len=target_len,
            euler_zyx_deg=euler_zyx_deg,
            root_pred_xyz=root_pred
        )

    out_path = os.path.splitext(input_bvh_path)[0] + "_pred_multi.bvh"
    with open(out_path, "w") as f:
        f.write(new_text)

    print(f"Saved predicted BVH to: {out_path}")

print("All predictions done")
