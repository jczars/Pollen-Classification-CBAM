import argparse
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import yaml

# Add the current directory to the PYTHONPATH
sys.path.insert(0, os.getcwd())
print(sys.path)

# Importing modules and functions
from models import grad_cam_lib as cam
from models import sound_test_finalizado


def load_imgs(path_data, images_labels, target_size=(224, 224)):
    images=[]
    for index in range(len(images_labels)):
        img_path = f"{path_data}{images_labels[index]}" 
        print(img_path)
        images.append(cam.load_img_gen(img_path, target_size, verbose=0))
    return images

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

# Função para extrair a classe
def extract_classes(file_paths):
    classes = [path.split("/")[0] for path in file_paths]
    return classes

def run(config):    
    path_model = config['path_model']
    path_data = config['path_data']
    # classes = config['classes']
    conv_layer_name = config['conv_layer_name']
    images_labels = config['images_labels']
    img_size = config['img_size']
    target_size = (img_size, img_size)
    nome = config['nome']
    
    images_list=[]
    print(images_labels)
    for index in range(len(images_labels)): 
        images_list.append(images_labels[index])

    classes_list = extract_classes(images_list)

    # classes_list=[]
    # for index in range(len(classes)): 
    #     classes_list.append(classes[index])

    print(f"[INFO] load model")
    model=cam.load_model(path_model)
    
    CATEGORIES = sorted(os.listdir(config['path_data']))
    print(f'[INFO] categories: {CATEGORIES}')
    images=load_imgs(path_data, images_list, target_size)

    fig=cam.display_cam_grid(images, classes_list, model, conv_layer_name, CATEGORIES)
    
    saved_dir = config['saved_dir']
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(saved_dir), exist_ok=True)

    image_saved = os.path.join(saved_dir, nome)
    fig.savefig(image_saved)

    print(f'[INFO] image saved: {image_saved}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data augmentation with specified configuration.")
    parser.add_argument("--config", type=str, default="./interpretation/config_class_well_k1.yaml", 
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load parameters from config file and process augmentation
    try:
        config = load_config(args.config)
        run(config)
        message = f'[INFO] finished successfully!!!'
        sound_test_finalizado.beep(2)
    except Exception as e:
        # Send error notification if the process fails            
        message = f'[INFO] with ERROR!!! {str(e)}'
        print(message)
        sound_test_finalizado.beep(2, message)