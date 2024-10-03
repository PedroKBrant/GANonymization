import os
import shutil

# Specify the path to the folder
folder_path = '/home/voxar/Desktop/pkb/datasets/MetaGaze_splitted/new_people/lena'

# List all files in the folder
files = os.listdir(folder_path)

# Iterate through each file and rename it
for filename in files:
    # Construct the new filename with "yuri_" prefix
    new_filename = f"lena_{filename}"
    
    # Rename the file
    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
    
    # Optionally, if using shutil.move() instead:
    # shutil.move(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
    
    print(f"Renamed {filename} to {new_filename}")

print("All files renamed successfully.")