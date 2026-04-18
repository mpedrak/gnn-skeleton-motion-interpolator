# Cuts a BVH file to keep frames in specic range

input_bvh_path = "./test/ACCAD/Male2_C18_RunToHopToWalk.bvh"
output_bvh_path = "./predict/ACCAD/male_run_to_hop_to_walk.bvh"
start_frame = 10
length = 100

# ---

import os

with open(input_bvh_path, 'r') as f:
    lines = f.readlines()

motion_idx = next(i for i, ln in enumerate(lines) if ln.strip().upper() == "MOTION")
n_frames = int(lines[motion_idx + 1].split(":")[1].strip())
frames_start_idx = motion_idx + 3

motion_lines = lines[frames_start_idx : frames_start_idx + n_frames]
trimmed_motion_lines = motion_lines[start_frame : start_frame + length]
lines[motion_idx + 1] = f"Frames: {len(trimmed_motion_lines)}\n"
new_lines = lines[ : frames_start_idx] + trimmed_motion_lines + lines[frames_start_idx + n_frames : ]

os.makedirs(os.path.dirname(output_bvh_path), exist_ok=True)

with open(output_bvh_path, 'w') as f:
    f.writelines(new_lines)

print(f"Trimmed BVH file saved to: {output_bvh_path}")
