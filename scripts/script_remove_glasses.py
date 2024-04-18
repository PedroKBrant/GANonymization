import pandas as pd
import os
from shutil import move

# Path to the CelebA dataset annotations file
annotations_file = 'lib/datasets/CelebA/labels.csv'
# Directory containing the CelebA images
images_dir = 'lib/datasets/CelebA/celeba/img/FacialLandmarks478/train'
# Directory to move the files with glasses
output_dir = 'lib/datasets/CelebA/celeba/img/FacialLandmarks478/train_only_glasses'

# Load the annotations file into a DataFrame
annotations_df = pd.read_csv(annotations_file)

# Filter out images with the "glasses" tag
filtered_df = annotations_df[annotations_df['Eyeglasses'] == 1]

# Get the list of image filenames to move
filtered_image_filenames = filtered_df['image_path'].tolist()

def move_files_from_list(images_dir, output_dir, file_list):
    for image_filename in os.listdir(images_dir):
        if image_filename.endswith('.jpg'):
            filename_parts = image_filename.split('-')
            last_part = filename_parts[-1].strip()
            print(last_part)
            if last_part in file_list:
                source_path = os.path.join(images_dir, image_filename)
                destination_path = os.path.join(output_dir, image_filename)
                print(images_dir, image_filename)
                print(output_dir, image_filename)
                move(source_path, destination_path)
            else:
                print(f"File not found: {last_part}")



move_files_from_list(images_dir, output_dir, filtered_image_filenames)
