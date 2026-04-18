# Splits BVH files into training and testing sets based on phrase in filename

data_dir = "./datasets/ACCAD"    

train_dir = "./train/ACCAD"   
train_phrases = ["Female1_B", "Female1_C", "Female1_D", "Male1_B", "Male1_C", "Male2_A", "Male2_B", "Male2_D", "Male2_E", "Male2_F", "Male2_G", "eric", "flip", "swagger"]

test_dir = "./test/ACCAD"   
test_phrases = ["Female1_A", "Male1_A", "Male2_C"]

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
