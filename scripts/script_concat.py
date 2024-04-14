from PIL import Image
import os

def concatenate_images(folder1, folder2, output_folder):
    # Get the list of files in both folders
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)
    print(files1)
    print("---------")
    print(files2)
    # Iterate through common files and concatenate images
    for file_name in files1:
        image1 = Image.open(os.path.join(folder1, file_name ))
        image2 = Image.open(os.path.join(folder2, file_name ))

        # Concatenate horizontally
        width, height = image1.size
        new_image = Image.new('RGB', (width * 2, height))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (width, 0))

        # Save concatenated image
        new_image.save(os.path.join(output_folder, file_name + '_concatenated.jpg'))

    print('Concatenation complete.')

# Example usage:
folder1 = 'results/melhoria_1/iris/after_2'
folder2 = 'results/melhoria_1/iris/before'
output_folder = 'results/melhoria_1/iris/concat'

concatenate_images(folder1, folder2, output_folder)
