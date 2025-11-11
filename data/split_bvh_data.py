# Splits BVH files into training and testing sets based on filename suffix

data_dir = "./lafan"    
suffix = "subject5.bvh"

# ---

import os
import shutil

train_dir = "./train"      
test_dir = "./test"        

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.lower().endswith(".bvh"):
        src_path = os.path.join(data_dir, filename)

        if filename.lower().endswith(suffix.lower()): dst_path = os.path.join(test_dir, filename)     
        else: dst_path = os.path.join(train_dir, filename)
            
        shutil.copy2(src_path, dst_path)
        print(f"Copied {filename} to {dst_path}")

print("Data splitting completed")
