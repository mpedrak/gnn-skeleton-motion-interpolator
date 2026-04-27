# Splits BVH files into training and testing sets based on phrase in filename

data_dir = "./datasets/SFU"   

test_dir = "./test/SFU-5"   
test_phrases = ["0005_BackwardsWalk001", "0008_Walking002", "0018_DanceTurns002", "0018_XinJiang003", "0015_JumpOverObstacle001"]

train_dir = "./train/SFU"   
train_phrases = ["_"]

# ---

import os
import shutil    

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.lower().endswith(".bvh"):
        src_path = os.path.join(data_dir, filename)

        if any(phrase in filename for phrase in test_phrases): dst_path = os.path.join(test_dir, filename)
        elif any(phrase in filename for phrase in train_phrases): dst_path = os.path.join(train_dir, filename)
        else: continue

        shutil.copy2(src_path, dst_path)
        print(f"Copied {filename} to {dst_path}")

print("Data splitting completed")
