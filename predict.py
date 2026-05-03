import os
import torch
import argparse
import json

from bvh import Bvh

from src.model import SkeletalMotionInterpolator
from src.utils.bvh import replace_gap_in_bvh_text, parse_bvh_file, get_bvh_frame_count, compress_skeleton_hierarchy
from src.utils.rotation import rot_6d_to_euler
from src.predict_gap import predict_gap
from src.utils.various import load_configs, set_global_seed


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
parser.add_argument("file", type=str)
parser.add_argument("gap_start", type=int) 
args = parser.parse_args()

input_bvh_path = args.file 
if not os.path.isfile(input_bvh_path):
    raise FileNotFoundError(f"Input BVH file not found: {input_bvh_path}")

filename = args.config
config, constants = load_configs([filename, "constants"])
print(f"Loaded config: {filename}")
print(f"Model description: {config['description']}")

gap_start_frame = args.gap_start - 1 # 0 based index in code
n_frames = get_bvh_frame_count(input_bvh_path)
if gap_start_frame <= config["context_len_pre"] or gap_start_frame >= n_frames - config["context_len_post"] - config["target_len"]:
    raise ValueError("Invalid gap start frame")        
        
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

set_global_seed(constants["seed"])


# Model
model = SkeletalMotionInterpolator(
    context_len_pre=config["context_len_pre"],
    context_len_post=config["context_len_post"],
    target_len=config["target_len"],
    rot_gnn_params=config["rot_gnn_params"],
    root_pos_mlp_params=config["root_pos_mlp_params"]
)
model = model.to(device)

model_path = constants["models_path"] + filename + constants["models_suffix"]
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
print(f"Loaded checkpoint: {model_path}")

skeletons_path = constants["skeletons_path"] + filename + constants["skeletons_suffix"]
with open(skeletons_path, 'r') as f:
    loaded_list = json.load(f)

model_supported_skeletons = {tuple(tuple(joint) for joint in skeleton) for skeleton in loaded_list}


# Parsing BVH 
with open(input_bvh_path, "r") as f:
    text = f.read()

mocap = Bvh(text)
root_pos, rot_6d, joint_names, parent_indices, offsets, rot_order = parse_bvh_file(input_bvh_path)
frames_total = rot_6d.shape[0]

compressed_skeleton = tuple(compress_skeleton_hierarchy(
    parent_indices=parent_indices, 
    joint_names=joint_names
))

if compressed_skeleton not in model_supported_skeletons:
    print(f"WARNING: this skeleton topology was not used during training")


# Prediction
context_len_pre = config["context_len_pre"]
context_len_post = config["context_len_post"]
target_len = config["target_len"]

model.eval()

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
        offsets=offsets
    )

euler_deg = rot_6d_to_euler(rot_6d=rot_pred, order=rot_order, degrees=True)

new_text = replace_gap_in_bvh_text(
    orig_text=text,
    mocap=mocap,
    gap_start=gap_start_frame,
    target_len=target_len,
    euler_deg=euler_deg,
    root_pred=root_pred
)

out_path = os.path.splitext(input_bvh_path)[0] + "_pred.bvh"
with open(out_path, "w") as f:
    f.write(new_text)

print(f"Saved predicted BVH to: {out_path}")
print(f"Replaced frames: [{gap_start_frame + 1}, {gap_start_frame + target_len}]")
print()
