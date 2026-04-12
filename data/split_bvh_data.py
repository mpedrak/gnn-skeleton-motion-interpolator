# Splits BVH files into training and testing sets based on filename suffixes

data_dir = "./lafan1"    

train_dir = "./train-small"   
train_suffixes = ["subject1.bvh"]

test_dir = "./test"   
test_suffixes = ["subject5.bvh"]

# ---

import os
import shutil    

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.lower().endswith(".bvh"):
        src_path = os.path.join(data_dir, filename)

        if filename.endswith(tuple(test_suffixes)): dst_path = os.path.join(test_dir, filename)
        elif filename.endswith(tuple(train_suffixes)): dst_path = os.path.join(train_dir, filename)
        else: continue

        shutil.copy2(src_path, dst_path)
        print(f"Copied {filename} to {dst_path}")

print("Data splitting completed")
