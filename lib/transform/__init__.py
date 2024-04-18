import os
import pathlib
import shutil
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger
from p_tqdm import p_tqdm
from tqdm import tqdm

from lib.transform.zero_padding_resize_transformer import ZeroPaddingResize
from lib.utils import glob_dir

def check_dimensions(images):
    # Get dimensions of the first image
    rows, cols, channels = images[0].shape

    # Check dimensions of all images
    for img in images[1:]:
        if img.shape != (rows, cols, channels):
            return True
    return False

def resize_to_match(image, target_image):
    return cv2.resize(image, (target_image.shape[1], target_image.shape[0]))

def exec_augmentation(files: List[str], output_dir: str, input_dir: str, size: int, gallery: bool,
                      transformer, mesh_configuration: str):
    """
    Executes the augmentation.
    @param files: The files to be augmented.
    @param output_dir: The directory of the output.
    @param input_dir: The directory of the input.
    @param size: The image size.
    @param gallery: Whether the image should be saved besides its original.
    @param transformer: The transformer to be applied.
    """
    logger.info(f"Parameters: {', '.join([f'{key}: {value}' for key, value in locals().items()])}")
    name = str(transformer)
    for image_file in files:
        sub_path_image = os.path.dirname(image_file[len(input_dir):])
        sub_output_dir = os.path.join(output_dir, *sub_path_image.split(os.sep))
        img = cv2.imread(image_file)
        if img is not None:
            if name == 'FacialLandmarks478':
                print('ENTREI AQUI FacialLandmarks478')
                print(mesh_configuration)
                pred = transformer(img, mesh_configuration)
            else:
                pred = transformer(img) 
            if not isinstance(pred, List):
                pred = [pred]
            for idx, sub_pred in enumerate(pred):
                sub_pred = ZeroPaddingResize(size)(sub_pred)
                os.makedirs(sub_output_dir, exist_ok=True)
                output_file = os.path.join(sub_output_dir,
                                           f'{name}_{idx}-{pathlib.Path(image_file).name}')
                if gallery:
                    if check_dimensions([img, sub_pred]) and img.dtype == sub_pred.dtype:
                        img = resize_to_match(img, sub_pred)
                    cv2.imwrite(output_file, cv2.hconcat([img, sub_pred]))
                else:
                    cv2.imwrite(output_file, sub_pred)
    return 0


def transform(input_dir: str, size: int, gallery: bool, transformer, mesh_configuration: str='00_pkb',
              output_dir: Optional[str] = None, num_workers: int = 1) -> str:
    """
    Transform all images found in the input directory.
    @param input_dir: The input directory.
    @param size: The size of the images afterward.
    @param gallery: Whether the image should be saved besides its original.
    @param transformer: The transformer to be applied.
    @param output_dir: The output directory.
    @param num_workers: The number of parallel workers.
    @return: The output path.
    """
    logger.info(f"Parameters: {', '.join([f'{key}: {value}' for key, value in locals().items()])}")
    name = str(transformer)
    if output_dir is None:
        output_dir = os.path.dirname(input_dir)
    output_dir = os.path.join(output_dir, name)
    logger.debug(f'Preparing output directory: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    # Copy other files to destination
    for f in os.listdir(input_dir):
        file = os.path.join(input_dir, f)
        if os.path.isfile(file):
            shutil.copyfile(file, os.path.join(output_dir, f))
    # Search for every image
    files = glob_dir(input_dir)
    logger.debug(f'Found {len(files)} files in source directory: {input_dir}')
    # Search for already processed files
    out_files = glob_dir(output_dir)
    out_files = [pathlib.Path(f).name.split('-')[-1] for f in out_files]
    logger.debug(f'Found {len(out_files)} files in destination directory: {output_dir}')
    files_augment = []
    for f in tqdm(files, desc='Skip Check'):
        if pathlib.Path(f).name.split('-')[-1] not in out_files:
            files_augment.append(f)
    if len(files_augment) > 0:
        logger.debug(f'Processing {len(files_augment)} files...')
        list_chunks = np.array_split(files_augment, num_workers)
        logger.debug(
            f'Distribute workload to {num_workers} workers with a chunk size of '
            f'{len(list_chunks[0])} each')
        p_tqdm.p_umap(exec_augmentation, list_chunks, [output_dir] * len(list_chunks),
                      [input_dir] * len(list_chunks),
                      [size] * len(list_chunks), [gallery] * len(list_chunks),
                      [transformer] * len(list_chunks),
                      num_cpus=num_workers, mesh_configuration = mesh_configuration)
    else:
        logger.debug('Data has already been fully processed!')
    return output_dir
