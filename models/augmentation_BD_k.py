# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os, glob, sys
import datetime
import imageio.v2 as imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import yaml

# Add the current directory to the PYTHONPATH
sys.path.insert(0, os.getcwd())

from models import sound_test_finalizado, utils 

def log_message(message, verbose):
    """Log a message if verbosity is enabled."""
    if verbose > 0:
        print(message)

def quantize_images(dst, categories, k_folds, verbose=0):
    """
    Quantizes the number of images in specified directories, displaying the count for each category.

    Parameters:
    - dst (str): Path to destination directory.
    - categories (list of str): List of category names (e.g., image classes).
    - k_folds (int): Number of folds to process.
    - verbose (int): If > 0, print additional information.
    """
    for i in range(k_folds):
        folder = os.path.join(dst, f'k{i+1}')
        log_message(f'Processing folder: {folder}', verbose)

        for category in categories:
            path = os.path.join(folder, category, '*.jpg')
            images = glob.glob(path)
            log_message(f'{category} - {len(images)} images', verbose)

def augment_images(images, factor, save_dir='', prefix='_Aug', img_type='png', verbose=0):
    """
    Applies augmentation to a batch of images.

    Parameters:
    - images (list): Array of images to augment.
    - factor (int): Number of augmentations to apply.
    - save_dir (str): Directory to save augmented images. If empty, does not save.
    - prefix (str): Prefix for augmented images.
    - img_type (str): Image format, default is 'png'.
    - verbose (int): If > 0, print additional information.

    Returns:
    - list: List of augmented images.
    """
    ia.seed(1)
    augment_seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=iap.Uniform(0.0, 0.3)),
        iaa.LinearContrast(iap.Choice([0.75, 1.25], p=[0.3, 0.7])),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
        iaa.Affine(rotate=(-25, 25), translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ], random_order=True)

    augmented_images = []
    for i in range(factor):
        augmented_batch = augment_seq(images=images)
        for idx, img in enumerate(augmented_batch):
            if save_dir:
                save_path = os.path.join(save_dir, f"{idx}_{prefix}.{img_type}")
                log_message(f"Saving augmented image to {save_path}", verbose)
                imageio.imwrite(save_path, img)
            augmented_images.append(img)

    log_message(f"Total augmented images: {len(augmented_images)}", verbose)
    return augmented_images

def calculate_augmentation(goal, num_images, images, img_type, save_dir='', verbose=0):
    """
    Calculates the number of augmentations required to meet a goal count of images.

    Parameters:
    - goal (int): Target number of images.
    - num_images (int): Current number of images.
    - images (list): List of images to augment.
    - img_type (str): Image format.
    - save_dir (str): Directory to save augmented images.
    - verbose (int): If > 0, print additional information.
    """
    if num_images < goal:
        difference = goal - num_images
        integer_part = difference // num_images
        fractional_part = difference % num_images

        if integer_part > 0:
            augment_images(images, integer_part, save_dir, prefix='_Aug', img_type=img_type, verbose=verbose)
        
        if fractional_part > 0:
            augment_images(images[:fractional_part], 1, save_dir, prefix='_AugFrac', img_type=img_type, verbose=verbose)

        log_message(f"Images needed: {goal}, current: {num_images}, additional required: {difference}", verbose)

def load_images_from_folder(path, img_type='png', verbose=0):
    """
    Loads all images of a specified type from a folder.

    Parameters:
    - path (str): Directory containing images.
    - img_type (str): Image format.
    - verbose (int): If > 0, print additional information.

    Returns:
    - np.array: Array of loaded images.
    """
    images = []
    image_paths = glob.glob(f"{path}/*.{img_type}")
    for img_path in image_paths:
        img = imageio.imread(img_path)
        images.append(img)

    log_message(f"Loaded {len(images)} images from {path}", verbose)
    return np.array(images)


def load_config(config_path="config.yaml"):
    """
    Loads configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: Dictionary with configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def log_message(message, verbose):
    if verbose > 0:
        print(message)


def load_images_from_folder(path, img_type, verbose=0):
    """
    Loads images from a specified folder.
    """
    images = []
    query = os.path.join(path, f'*.{img_type}')
    image_paths = glob.glob(query)
    for img_path in image_paths:
        img = imageio.imread(img_path)
        images.append(img)
    log_message(f"Loaded {len(images)} images from {path}", verbose)
    return np.array(images)


def calculate_augmentation(goal, current_count, images, img_type, save_dir, verbose=0):
    """
    Augments images to reach the specified goal count.
    """
    needed = goal - current_count
    if needed <= 0:
        log_message("No augmentation needed; goal already met.", verbose)
        return

    full_batches = needed // current_count
    partial_batch = needed % current_count

    aug_images = augment_images(images, full_batches, save_dir, img_type, verbose)
    if partial_batch > 0:
        aug_images += augment_images(images[:partial_batch], 1, save_dir, img_type, verbose)

    log_message(f"Generated {len(aug_images)} augmented images for {save_dir}", verbose)


def augment_images(images, factor, save_dir, img_type='png', verbose=0):
    """
    Applies data augmentation to images.
    """
    aug = iaa.Sequential([
        iaa.GaussianBlur(sigma=iap.Uniform(0.0, 0.3)),
        iaa.contrast.LinearContrast(iap.Choice([0.75, 1.25], p=[0.3, 0.7])),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
        iaa.Affine(rotate=(-25, 25), translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ], random_order=True)

    aug_images = []
    for _ in range(factor):
        augmented_batch = aug(images=images)
        for img in augmented_batch:
            filename = os.path.join(save_dir, f"aug_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.{img_type}")
            imageio.imwrite(filename, img)
            aug_images.append(img)
            log_message(f"Saved augmented image: {filename}", verbose)
    return aug_images


def run_balancing(dst_base, categories, goal, img_type, k_folds, save_dir, verbose=0):
    """
    Main function to balance image datasets by augmenting images to reach a specified goal.
    """
    start_time = datetime.datetime.now()
    log_data = [["algorithm", "base", "start", "end", "duration", "date"]]

    for i in range(k_folds):
        fold_path = os.path.join(dst_base, f'Train/k{i+1}')
        log_message(f"Processing fold: {fold_path}", verbose)

        for category in categories:
            category_path = os.path.join(fold_path, category)
            images = load_images_from_folder(category_path, img_type, verbose)
            calculate_augmentation(goal, len(images), images, img_type, 
                                   save_dir=category_path, verbose=verbose)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    log_data.append(["run_balancing", dst_base, start_time, end_time, duration, datetime.date.today()])

    utils.add_row_csv(os.path.join(save_dir, f"balance_log_{datetime.date.today()}.csv"), log_data)
    log_message("Balancing complete. Log saved.", verbose)


def process_augmentation(config):
    """
    Processes data augmentation for each view specified in the configuration.

    Parameters:
    - config (dict): Dictionary with configuration parameters.
    """
    type = config['type']
    goal = config['goal']
    k_folds = config['k_folds']
    base_dir = config['base_dir']
    vistas = config['vistas']
    verbose = config['verbose']


    
    # Process each view specified in the YAML
    for vista in vistas:
        dst_base = os.path.join(base_dir, f'{vista}_R')
        save_dir = base_dir
        
        # Determine categories based on folders in the dataset
        bd_src = os.path.join(dst_base, 'Train/k1')
        categories = sorted(os.listdir(bd_src))
        
        # Run the main balancing function
        run_balancing(dst_base, categories, goal, type, k_folds, save_dir, verbose=verbose)
        
        if verbose > 0:
            print(f"Data augmentation completed for view: {vista}")

    print('Data augmentation process completed for all views.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data augmentation with specified configuration.")
    parser.add_argument("--config", type=str, default="./preprocess/config_balanced.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load parameters from config file and process augmentation
    config = load_config(args.config)
    process_augmentation(config)
    sound_test_finalizado.beep(2)

