# Deletes cache files from selected directories based on suffixes

dirs_to_clear = ["./train-small", "./test", "./train"]
suffixes_to_clear = [".pt"]

# ---

import os

for dir_path in dirs_to_clear:
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            if any(filename.lower().endswith(suffix.lower()) for suffix in suffixes_to_clear):
                file_path = os.path.join(dir_path, filename)
                os.remove(file_path)
                print(f"Deleted {file_path}")
    else:
        print(f"Directory {dir_path} does not exist")

print("Cache clearing completed")
