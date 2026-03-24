import os
import torch
import numpy as np
import argparse

from bvh import Bvh

from src.model import SkeletalMotionInterpolator
from src.utils.bvh import replace_gap_in_bvh_text, parse_bvh_file, get_bvh_frame_count
from src.utils.rotation import rot_6d_to_euler_zyx
from src.predict_gap import predict_gap
from src.utils.various import load_configs


# Additional arguments
predict_data_dir = "./data/predict/"
bvh_files = ["aiming", "dance", "fight", "ground", "jumps", "run", "walk"]
hole_starts = [i for i in range(20, 480 + 1, 20)]


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()
filename = args.config

config, constants = load_configs([filename, "constants"])
print(f"Loaded config: {filename}")
print(f"Model description: {config['description']}")
        
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


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


# Prediction
context_len_pre = config["context_len_pre"]
context_len_post = config["context_len_post"]
target_len = config["target_len"]

model.eval()

for bvh_file in bvh_files:
    # Parsing BVH 
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

        # Interpolation
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
                gap_start=gap_start_frame
            )

        euler_zyx_deg = rot_6d_to_euler_zyx(rot_pred, degrees=True)

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
print()
