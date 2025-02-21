# Add the current directory to the PYTHONPATH
import argparse
import os
import sys

import yaml


sys.path.insert(0, os.getcwd())
print(sys.path)

from models import reports_gen_view as reports 
from models import sound_test_finalizado

def run(params):
    print(f"params: {params}")
    

def parse_args():
    """
    Parse command-line arguments for the script.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Resize images and organize them by category.")
    
    parser.add_argument('--config', type=str, help="Path to the configuration YAML file.")
    
    return parser.parse_args()

def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    args = parse_args()

    # Load configuration from YAML file
    config_file = args.config if args.config else 'src/config_test_vw.yaml'
    params = load_config(config_file)

    #run(params)
    #python 1_create_bd/separeted_bd.py --config 1_create_bd/config_separeted.yaml
    
    debug = True
    
    if debug:
        # Run the training and evaluation process in debug mode
        run(params)
        sound_test_finalizado.beep(2)
    else:        
        try:
            # Run the training and evaluation process and send success notification
            run(params)
            message = '[INFO] successfully!'
            print(message)
            sound_test_finalizado.beep(2, message)
        except Exception as e:
            # Send error notification if the process fails            
            message = f'[INFO] with ERROR!!! {str(e)}'
            print(message)
            sound_test_finalizado.beep(2, message)