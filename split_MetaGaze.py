import os
import random
import shutil

def split_dataset(image_dir, output_dir, folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Check if the ratios sum to 1
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("Train, validation, and test ratios must sum to 1")

    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all image files
    images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # Shuffle images
    random.shuffle(images)

    # Compute split sizes
    total_images = len(images)
    train_size = int(total_images * train_ratio)
    val_size = int(total_images * val_ratio)
    test_size = total_images - train_size - val_size

    # Split images
    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    # Copy images to corresponding directories
    for img in train_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(train_dir, folder+'_'+img))

    for img in val_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(val_dir, folder+'_'+img))

    for img in test_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(test_dir, folder+'_'+img))

    print(f'Total images: {total_images}')
    print(f'Training images: {train_size}')
    print(f'Validation images: {val_size}')
    print(f'Test images: {test_size}')
    print('Dataset split completed!')

# Example usage
image_dir = '../MetaGaze/'
output_dir = '../MetaGaze_splitted/'
folders = [name for name in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, name))]
folders.sort()

for folder in folders:
    print(image_dir+'/'+folder)
    split_dataset(image_dir+'/'+folder+'/', output_dir, folder)