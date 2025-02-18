import sys
import os
import cv2
import glob
import argparse
import yaml
from tqdm import tqdm

# Add custom module path to system path
#sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
#print("System paths:", sys.path)

# Import custom modules and functions
from models import utils, sound_test_finalizado

def save_resize(src, dst, file_type='jpg', save=0, verbose=0):
    """
    Resize and save images from source to destination directory.

    Parameters:
    - src (str): Source directory with images.
    - dst (str): Destination directory where resized images will be saved.
    - file_type (str): Type of images to process (e.g., 'jpg', 'png'). Default is 'jpg'.
    - save (int): Whether to save resized images (1 to save, 0 to not save). Default is 0.
    - verbose (int): Verbosity level (0 for silent, 1 for detailed messages). Default is 0.
    """
    # Iterate over images in the source directory
    for root, dirs, files in tqdm(os.walk(src)):
        for filename in files:
            if filename.endswith(f".{file_type}"):
                src_img_path = os.path.join(root, filename)
                dst_img_path = os.path.join(dst, filename)

                # Print details if verbose > 0
                if verbose > 0:
                    print(f'\n[STEP 1 * ] Processing {filename}')
                    print(f'Source: {src_img_path}')
                    print(f'Destination: {dst_img_path}')

                # Resize and save the image if requested
                if save == 1:
                    print(f'[STEP 2 *] Saving resized image...')
                    img = cv2.imread(src_img_path)
                    img_re = cv2.resize(img, (224, 224))
                    cv2.imwrite(dst_img_path, img_re)

    # Print total number of images processed if verbose > 0
    if verbose > 0:
        images_path = glob.glob(os.path.join(src, f"*.{file_type}"))
        print(f'Total images in source directory: {len(images_path)}')
        images_path = glob.glob(os.path.join(dst, f"*.{file_type}"))
        print(f'Total images in destination directory: {len(images_path)}')


def run(params):
    """
    Run the image processing pipeline, iterating over categories and resizing images.

    Parameters:
    - params (dict): A dictionary with the following keys:
        - 'type': Image file type (e.g., 'png', 'jpg').
        - 'bd_src': Source directory for images.
        - 'bd_dst': Destination directory for resized images.
        - 'save_dir': Directory to save the graph image.
        - 'save': Flag to save resized images (1 = save, 0 = don't save).
        - 'verbose': Verbosity level (1 = show details, 0 = silent).
    """
    categories = sorted(os.listdir(params['bd_src']))

    utils.create_folders(params['bd_dst'], flag=1)

    for category in categories:
        src = os.path.join(params['bd_src'], category)
        dst = os.path.join(params['bd_dst'], category)

        utils.create_folders(dst, 0)

        if params['verbose'] == 0:
            print(f'\nSource: {src}')
            print(f'Destination: {dst}')

        save_resize(src, dst, params['type'], params['save'], params['verbose'])

    fig = utils.graph_img_cat(params['bd_dst'])
    save_dir = params['save_dir']
    sp=params['bd_src'].split('/')[-1]
    nm_img = f'{sp}_quant.jpg'
    if save_dir:
        fig.savefig(os.path.join(save_dir, nm_img))


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
    config_file = args.config if args.config else './preprocess/config_resize.yaml'
    params = load_config(config_file)

    #run(params)
    #python resize_r1.py --config 1_create_bd/config_resize.yaml
    
    debug = False
    
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
