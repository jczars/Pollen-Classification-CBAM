# -*- coding: utf-8 -*-

# Libraries
import argparse
import os
from math import ceil
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil, glob
from sklearn.model_selection import StratifiedKFold
import yaml

def load_config(config_path="config.yaml"):
    """
    Loads configuration parameters from a YAML file.
    
    Parameters:
    - config_path (str): Path to the YAML configuration file.
    
    Returns:
    - dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_dataset(data_path, csv_path, categories):
    """
    Creates a dataset in CSV format with file paths and associated labels.
    
    Parameters:
    - data_path (str): Path to the root directory containing the data.
    - csv_path (str): Path to save the generated CSV file.
    - categories (list): List of categories (subfolders) in the data directory.
    
    Returns:
    - DataFrame: DataFrame with file paths and labels.
    """
    data = pd.DataFrame(columns=['file', 'labels'])
    c = 0

    for category in tqdm(categories, desc="Processing categories"):
        category_path = os.path.join(data_path, category)

        # Check if path exists and is a directory
        if not os.path.exists(category_path) or not os.path.isdir(category_path):
            print(f"Warning: {category_path} is not a valid directory. Skipping...")
            continue

        filenames = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f)) and not f.startswith(".")]

        for filename in filenames:
            data.loc[c] = [os.path.join(category_path, filename), category]
            c += 1

    # Save the data to CSV
    data.to_csv(csv_path, index=False, header=True)
    data_csv = pd.read_csv(csv_path)

    print(f'\nCSV data saved to: {csv_path}')
    print(data_csv.groupby('labels').count())
    return data

def create_folders(save_dir, overwrite_flag=1):
    """
    Creates a directory if it doesn't already exist.
    
    Parameters:
    - save_dir (str): Path to the folder.
    - overwrite_flag (int): If 1, raises an error if the folder exists. If 0, prints a message instead.
    """
    if os.path.isdir(save_dir):
        if overwrite_flag:
            raise FileNotFoundError(f"Folder already exists: {save_dir}")
        else:
            print(f"Folder already exists: {save_dir}")
    else:
        os.mkdir(save_dir)
        print(f"Folder created: {save_dir}")

def kfold_split(data_csv, save_path, k_folds, base_name):
    """
    Splits the dataset into K folds (train/test).
    
    Parameters:
    - data_csv (DataFrame): The dataset to split.
    - save_path (str): Path to save the split CSV files.
    - k_folds (int): Number of folds for cross-validation.
    - base_name (str): Base name for the CSV files.
    """
    Y = data_csv[['labels']]
    n = len(Y)
    print(f"Total number of data points: {n}")
    kfold = StratifiedKFold(n_splits=int(k_folds), random_state=7, shuffle=True)

    k = 1
    for train_index, test_index in kfold.split(np.zeros(n), Y):
        train_data = data_csv.iloc[train_index]
        test_data = data_csv.iloc[test_index]

        print(f"Train set size: {len(train_index)}")
        print(f"Test set size: {len(test_index)}")
        print(f"Fold {k}")

        # Save split data to CSV
        train_csv_path = os.path.join(save_path, f"{base_name}_trainSet_k{k}.csv")
        test_csv_path = os.path.join(save_path, f"{base_name}_testSet_k{k}.csv")

        train_data.to_csv(train_csv_path, index=False, header=True)
        test_data.to_csv(test_csv_path, index=False, header=True)

        print(train_data.groupby('labels').count())
        k += 1

def copy_images(training_data, dst):
    """
    Copies images from the source to the destination directory.
    
    Parameters:
    - training_data (DataFrame): DataFrame with file paths to the images.
    - dst (str): Destination directory where the images will be copied.
    """
    for file_path in training_data['file']:
        folder = file_path.split('/')[-2]
        filename = file_path.split('/')[-1]

        dst_folder = os.path.join(dst, folder)
        create_folders(dst_folder, overwrite_flag=0)

        dst_file = os.path.join(dst_folder, filename)
        shutil.copy(file_path, dst_file)

def copy_images_for_k_folds(csv_path, train_path, base_name, k_folds, set_type='train'):
    """
    Copies images for each fold based on the CSV file.
    
    Parameters:
    - csv_path (str): Path to the CSV file containing file paths.
    - train_path (str): Path to the train or test directory.
    - base_name (str): Base name used to construct CSV paths.
    - k_folds (int): Number of folds.
    - set_type (str): 'train' or 'test' set.
    """
    for i in range(k_folds):
        k = i + 1
        folder = os.path.join(train_path, f'k{k}')
        create_folders(folder, overwrite_flag=0)

        path_csv = os.path.join(csv_path, f"{base_name}_{set_type}Set_k{k}.csv")
        data = pd.read_csv(path_csv)
        copy_images(data, folder)

def quantize_images(dst, categories, k_folds):
    """
    Quantizes images by counting them in each category and fold.
    
    Parameters:
    - dst (str): Destination directory where images are stored.
    - categories (list): List of categories.
    - k_folds (int): Number of folds.
    """
    for i in range(k_folds):
        k = i + 1
        folder = os.path.join(dst, f'k{k}')
        print(f"Folder: {folder}")

        for category in categories:
            category_path = os.path.join(folder, category, "*.png")
            images_path = glob.glob(category_path)
            print(f"{category}: {len(images_path)} images")

def copy_csv_files(src, dst_csv):
    """
    Copies all CSV files from the source to the destination directory.
    
    Parameters:
    - src (str): Source directory where CSV files are located.
    - dst_csv (str): Destination directory where CSV files will be copied.
    """
    csv_files = glob.glob(os.path.join(src, "*.csv"))
    print(f"CSV files: {csv_files}")

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        dst_file = os.path.join(dst_csv, filename)
        shutil.copy(csv_file, dst_file)

def create_folder_structure(base_path):
    """
    Creates folder structure for training and testing data.
    
    Parameters:
    - base_path (str): Base path where the folder structure will be created.
    """
    create_folders(base_path, overwrite_flag=0)

    train_folder = os.path.join(base_path, 'Train')
    create_folders(train_folder, overwrite_flag=0)

    test_folder = os.path.join(base_path, 'Test')
    create_folders(test_folder, overwrite_flag=0)

    csv_folder = os.path.join(base_path, 'csv')
    create_folders(csv_folder, overwrite_flag=0)

def copy_data(base_src, base_dst, base_name, categories, k_folds):
    """
    Copies the data and prepares the dataset for training and testing.
    
    Parameters:
    - base_src (str): Source directory with the original data.
    - base_dst (str): Destination directory where the new dataset will be stored.
    - base_name (str): Base name for dataset files.
    - categories (list): List of categories in the data.
    - k_folds (int): Number of k-folds.
    """
    print("Creating folder structure...")
    create_folder_structure(base_dst)

    print("Creating dataset CSV...")
    csv_path = os.path.join(base_dst, 'csv')
    csv_file = os.path.join(csv_path, f"{base_name}.csv")
    data_csv = create_dataset(base_src, csv_file, categories)

    print("Splitting data into k-folds...")
    kfold_split(data_csv, csv_path, k_folds, base_name)

    print("Copying images to Train folder...")
    train_path = os.path.join(base_dst, 'Train')
    copy_images_for_k_folds(csv_path, train_path, base_name, k_folds, set_type='train')
    quantize_images(train_path, categories, k_folds)

    print("Copying images to Test folder...")
    test_path = os.path.join(base_dst, 'Test')
    copy_images_for_k_folds(csv_path, test_path, base_name, k_folds, set_type='test')
    quantize_images(test_path, categories, k_folds)

def run_split(params):
    """
    Runs the data preparation process based on the provided parameters.
    
    Parameters:
    - params (dict): Configuration parameters.
    """
    base_dir = params['base_dir']
    views = params['views']
    k_folds = params['k_folds']
    goal = params['goal']

    base_path = os.path.dirname(os.path.dirname(base_dir))
    base_name = os.path.basename(os.path.normpath(base_dir))
    new_base_name = f"{base_name}_{goal}"
    new_base_dir = os.path.join(base_path, new_base_name)

    if views == 'None':
        categories = sorted(os.listdir(base_dir))
        copy_data(base_dir, new_base_dir, new_base_name, categories, k_folds)
    else:
        for view in views:
            view_path = os.path.join(base_dir, view)
            categories = sorted(os.listdir(view_path))

            view_dst = os.path.join(base_dir, f"{view}_{goal}")
            create_folders(view_dst, overwrite_flag=1)
            copy_data(view_path, view_dst, base_name, categories, k_folds)
    return  new_base_dir

if __name__ == "__main__":
    # Argument parser for command-line configuration
    parser = argparse.ArgumentParser(description="Run the script with YAML configuration.")
    parser.add_argument(
        "--config",
        type=str,
        default="./preprocess/config_balanced_10.yaml",
        help="Path to the YAML configuration file. Defaults to 'config.yaml'."
    )
    args = parser.parse_args()
    params = load_config(args.config)
    run_split(params)
