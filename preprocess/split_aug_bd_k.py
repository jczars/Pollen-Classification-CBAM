import argparse
import yaml
import os, sys

os.environ["tf_gpu_allocator"]="cuda_malloc_async"
# Add the current directory to the PYTHONPATH
sys.path.insert(0, os.getcwd())
print(sys.path)

from models import split_BD_Views_k, augmentation_BD_k, sound_test_finalizado

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
    views = config['views']
    verbose = config['verbose']

    new_base_dir=split_BD_Views_k.run_split(config)

    if views == 'None':
        # Determine categories based on folders in the dataset
        bd_src = os.path.join(new_base_dir, 'Train/k1')
        categories = sorted(os.listdir(bd_src))
        augmentation_BD_k.run_balancing(new_base_dir, categories, goal, type, k_folds, new_base_dir, verbose=verbose)

    else:
        # Process each view specified in the YAML
        for view in views:
            dst_base = os.path.join(base_dir, f'{view}_{goal}')
            save_dir = base_dir
            
            # Determine categories based on folders in the dataset
            bd_src = os.path.join(dst_base, 'Train/k1')
            categories = sorted(os.listdir(bd_src))
            
            # Run the main balancing function
            augmentation_BD_k.run_balancing(dst_base, categories, goal, type, k_folds, save_dir, verbose=verbose)
            
            if verbose > 0:
                print(f"Data augmentation completed for view: {view}_{goal}")

    print('Data augmentation process completed for all views.')

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data augmentation with specified configuration.")
    parser.add_argument("--config", type=str, default="preprocess/config_split_aug_B73.yaml", 
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load parameters from config file and process augmentation
    #python3 preprocess/aug_balanc_bd_k.py --config preprocess/config_balanced.yaml
    config = load_config(args.config)
    process_augmentation(config)
    sound_test_finalizado.beep(2)