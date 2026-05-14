# Splits BVH files into training and testing sets based on phrase in filename

data_dir = "./datasets/UNOC_30"   

test_dir = "./test/UNOC-S2-S6-S9"   
test_phrases = ["S2", "S6", "S9"]

train_dir = "./train/UNOC"   
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
