
import shutil
import os, sys
import numpy as np
import pandas as pd
from keras import models
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from keras import models
import  argparse, yaml

# Adiciona o diretório atual (pwd) ao PYTHONPATH
sys.path.insert(0, os.getcwd())
# Import custom modules and functions
from models import get_data, utils, sound_test_finalizado

def initial(params, k):
    
    # Load categories from the source directory
    bd_src=f"{params['bd_src']}/k{k}"
    #bd_src = os.path.join(params['bd_src'], f'/k{k}')
    print(f"bd_src: {bd_src}")
        
    # Ensure destination folder exists
    bd_dst=f"{params['bd_dst']}/k{k}"
    print(f"bd_dst: {bd_dst}")
    os.makedirs(bd_dst, exist_ok=True)        

    return bd_src, bd_dst

def create_dataSet(bd_src, bd_dst):

    data = pd.DataFrame(columns=['file', 'labels'])
    c = 0
    cat_names = os.listdir(bd_src)
    for j in tqdm(cat_names, desc="Processing categories"):
        pathfile = os.path.join(bd_src, j)
        # Check if the path is a directory
        if not os.path.isdir(pathfile):
            print(f"Warning: {pathfile} is not a directory, skipping.")
            continue

        filenames = os.listdir(pathfile)
        for i in filenames:
            # Full file path
            file_path = os.path.join(pathfile, i)
            
            # Check if it's a valid file (e.g., image file)
            if os.path.isfile(file_path):
                data.loc[c] = [file_path, j]
                c += 1

    _csv_data=f"{bd_dst}data.csv"
    data.to_csv(_csv_data, index=False, header=True)
    print(f'\nCSV saved successfully at: {_csv_data}')

    return _csv_data

def read_data_csv(_csv_data):
    data_csv = pd.read_csv(_csv_data)        
    # Create and save the summary CSV with counts of images per label
    _summary_csv = _csv_data.replace('.csv', '_summary.csv')
    label_counts = data_csv.groupby('labels').size().reset_index(name='count')
    label_counts.to_csv(_summary_csv, index=False, header=True)
    print(data_csv.groupby('labels').count())
    
    return data_csv

def predict_data_generator(test_data_generator, model, categories, batch_size, verbose=2):
    filenames = test_data_generator.filenames
    df = pd.DataFrame(filenames, columns=['file'])
    nb_samples = len(filenames)
    print('Predicting unlabeled data...', nb_samples)
    print(f'Batch size: {batch_size}')
    y_preds = model.predict(test_data_generator)
    y_pred = np.argmax(y_preds, axis=1)
    df['y_pred'] = y_pred

    vistas = []   # List to store views
    classes = []  # List to store classes

    # Iterar sobre as linhas do DataFrame
    for i, row in df.iterrows():
        # Access category prediction and file path
        vista = categories[row['y_pred']]  # Corrigido para acessar 'y_pred' da linha atual
        classe = row['file']

        # Extract the view ("EQUATORIAL" or "POLAR") and class from the file path
        vt = vista.split('_')[0]        
        classe = classe.split('/')[-2]    

        vistas.append(vt)
        classes.append(classe)

    df['vista'] = vistas
    df['classe'] = classes

    # Group DataFrame by 'vista' and 'classe' and count the number of images per combination
    quantidade_por_vista_classe = df.groupby(['vista', 'classe']).size().reset_index(name='quantidade')

    return df, quantidade_por_vista_classe

def copy_images_by_vista(bd_dst, df):
    # Ensure the destination directories exist
    equatorial_dir = f"{bd_dst}/EQUATORIAL" 
    print(f'equatorial_dir: {equatorial_dir}')
    
    polar_dir = f"{bd_dst}/POLAR"
    print(f'polar_dir: {polar_dir}')

    os.makedirs(equatorial_dir, exist_ok=True)
    os.makedirs(polar_dir, exist_ok=True)

    # Iterate through the DataFrame and copy images based on the 'vista' column
    for _, row in df.iterrows():
        # Get file path and vista from DataFrame
        file_path = row['file']  # full path of the image
        vista = row['vista']

        # Extract class from the file path (second-to-last directory in the path)
        class_name = file_path.split('/')[-2]  # Class is the second-to-last element in the path

        # Determine the destination folder based on vista
        if vista == 'equatorial':
            destination_folder = os.path.join(equatorial_dir, class_name)
        elif vista == 'polar':
            destination_folder = os.path.join(polar_dir, class_name)
        else:
            continue  # Skip if vista is not valid

        # Ensure the class subdirectory exists
        os.makedirs(destination_folder, exist_ok=True)

        # Determine the destination file path
        destination_path = os.path.join(destination_folder, os.path.basename(file_path))

        # Copy the image to the appropriate directory
        try:
            shutil.copy(file_path, destination_path)
            print(f"Copied {file_path} to {destination_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error copying {file_path}: {e}")

def run(params):
    print(f"params: {params}")
    image_size=params['image_size'] 
    input_shape=(image_size, image_size)
    # Load categories from the labels directory
    categories_vistas = sorted(os.listdir(params['path_labels']))
    # Load the model for predictions
    print(f"\n[INFO] load model")
    model = models.load_model(params['path_model'])
    model.summary()  # Print model summary

    for i in range(10):
        k=i+1
        print(f"k={k}")

        print(f"\n[INFO] initial")
        bd_src, bd_dst = initial(params, k)

        print(f"\n[INFO] create dataset")
        _csv_data=create_dataSet(bd_src, bd_dst) 

        print(f"\n[INFO] read dataset")
        data=read_data_csv(_csv_data)
        print(f"data: {data}")
        
        print(f"\n[INFO] predict")
        test_data_generator = get_data.load_data_test(data, input_shape) 
        df_vistas, df_quantidade =predict_data_generator(test_data_generator, model, 
                                                         categories_vistas, params['batch_size'], verbose=2)

        #df_vistas.to_csv(f"{bd_dst}df_vistas.csv", index=False)
        #df_quantidade.to_csv(f"{bd_dst}df_qde_vistas.csv", index=False)

        copy_images_by_vista(bd_dst, df_vistas)

    print(f"\n[INFO] finished")


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
    config_file = args.config if args.config else 'phase3/config_separeted.yaml'
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