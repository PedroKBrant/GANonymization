import os

folder_path = "lib/datasets/CelebA/celeba/img"

# List all files in the folder
files = os.listdir(folder_path)

# Filter files that end with '.jpg'
jpg_files = [file for file in files if file.lower().endswith(".jpg")]

# Delete each jpg file
for jpg_file in jpg_files:
    file_path = os.path.join(folder_path, jpg_file)
    os.remove(file_path)
    print(f"Deleted: {jpg_file}")