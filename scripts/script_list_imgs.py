
import os
import shutil

def create_txt_file(folder_path, txt_filename):
    with open(txt_filename, 'w') as txt_file:
        for filename in os.listdir(folder_path):
            txt_file.write(filename + '\n')

def copy_files_from_txt(txt_filename, source_folder, destination_folder):
    with open(txt_filename, 'r') as txt_file:
        for line in txt_file:
            print(line)
            filename_part = line.strip()
            source_path = None
            # Find the file in the source folder
            for filename in os.listdir(source_folder):
                if filename.endswith(filename_part):
                    source_path = os.path.join(source_folder, filename)
                    break

            # If file found, copy it to the destination folder
            if source_path is not None:
                destination_path = os.path.join(destination_folder, filename)
                shutil.copy2(source_path, destination_path)




folder_path = '/home/voxar/Desktop/pkb/GANonymization/relatorio/03_experiment/results/00_pkb'
txt_filename = 'elipse.txt'
create_txt_file(folder_path, txt_filename)


source_folder = '/home/voxar/Desktop/pkb/GANonymization/lib/datasets/CelebA/celeba/img/FaceSegmentation/val'
destination_folder = '/home/voxar/Desktop/pkb/GANonymization/relatorio/03_experiment/input'

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

copy_files_from_txt(txt_filename, source_folder, destination_folder)






