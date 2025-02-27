import argparse
import os, sys
import yaml

# Add the current directory to the PYTHONPATH
sys.path.insert(0, os.getcwd())
print(sys.path)

from models import  consolidated_boxplot, consolidated_matrix, sound_test_finalizado

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

def run(config):

    k_folds = config['k_folds']
    folders = config['folder']
    normalize = config['normalize']

    # Process each view specified in the YAML
    for folder in folders:
        print(folder)
        consolidated_matrix.run(folder, normalize)
        consolidated_boxplot.run(folder, k_folds)


if __name__ == "__main__":
    """
    folder = './results/phase2/reports_cr_13_500/0_DenseNet201_reports/'    
    k=10
    normalize=True
    run(folder, k, normalize)
    """
    parser = argparse.ArgumentParser(description="Run data augmentation with specified configuration.")
    parser.add_argument("--config", type=str, default="./discussion/config_consolidaded.yaml", 
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load parameters from config file and process augmentation
    #python3 preprocess/aug_balanc_bd_k.py --config preprocess/config_balanced.yaml
    config = load_config(args.config)
    run(config)
    sound_test_finalizado.beep(2)