# Downsamples all BVH files in a directory (e.g. 120Hz -> 30Hz is done by keeping every 4th frame and new frame time is 0.033333 (1/30))

input_dir = "./datasets/100STYLE_60/"
output_dir = "./datasets/100STYLE_30/"
keep_every = 2
new_frame_time = 0.033333

# ---

import os

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith('.bvh'): continue
        
    input_bvh_path = os.path.join(input_dir, filename)
    output_bvh_path = os.path.join(output_dir, filename)
    
    with open(input_bvh_path, 'r') as f:
        lines = f.readlines()

    motion_idx = next(i for i, ln in enumerate(lines) if ln.strip().upper() == "MOTION")
    n_frames = int(lines[motion_idx + 1].split(":")[1].strip())
    frames_start_idx = motion_idx + 3

    motion_lines = lines[frames_start_idx : frames_start_idx + n_frames]
    downsampled_motion_lines = motion_lines[::keep_every]
    
    lines[motion_idx + 1] = f"Frames: {len(downsampled_motion_lines)}\n"
    lines[motion_idx + 2] = f"Frame Time: {new_frame_time}\n"
    
    new_lines = lines[ : frames_start_idx] + downsampled_motion_lines + lines[frames_start_idx + n_frames : ]

    with open(output_bvh_path, 'w') as f:
        f.writelines(new_lines)

    print(f"Downsampled {filename} to {output_bvh_path}")

print("Downsampling completed")
