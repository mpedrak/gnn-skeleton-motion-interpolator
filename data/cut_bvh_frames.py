# Cuts a BVH file to keep frames in specic range

input_bvh_path = "./test/run1_subject5.bvh"
output_bvh_path = "./predict/test_3.bvh"
start_frame = 50
length = 200

# ---

with open(input_bvh_path, 'r') as f:
    lines = f.readlines()

motion_idx = next(i for i, ln in enumerate(lines) if ln.strip().upper() == "MOTION")
n_frames = int(lines[motion_idx + 1].split(":")[1].strip())
frames_start_idx = motion_idx + 3

motion_lines = lines[frames_start_idx : frames_start_idx + n_frames]
trimmed_motion_lines = motion_lines[start_frame : start_frame + length]
lines[motion_idx + 1] = f"Frames: {len(trimmed_motion_lines)}\n"
new_lines = lines[ : frames_start_idx] + trimmed_motion_lines + lines[frames_start_idx + n_frames : ]

with open(output_bvh_path, 'w') as f:
    f.writelines(new_lines)

print(f"Trimmed BVH file saved to: {output_bvh_path}")
