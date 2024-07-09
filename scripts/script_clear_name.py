import os

# Specify the directory where your files are located
directory = '/home/voxar/Desktop/pkb/GANonymization/lib/datasets/pkb_dataset/FaceSegmentation'

# Iterate through each file in the directory
for filename in os.listdir(directory):
    # Split the filename by '-' to get the parts
    #parts = filename.split('-')
    
    # Extract the part after the last '-' (assuming there is at least one '-')
    #new_filename = parts[-1]
    new_filename = filename[-9:]
    # Construct the new path with the new filename
    new_path = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(os.path.join(directory, filename), new_path)
    
    # Print the old and new filenames for verification
    print(f'Renamed: {filename} -> {new_filename}')