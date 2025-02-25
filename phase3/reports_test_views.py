# Add the current directory to the PYTHONPATH
import argparse
import csv
import os
import sys

import openpyxl
import pandas as pd
import yaml
from keras import models

sys.path.insert(0, os.getcwd())
print(sys.path)

from models import maneger_gpu, reports_gen_view as reports 
from models import sound_test_finalizado

def reset_environment():
    
    """
    Resets the computing environment by clearing the Keras backend session.

    This function is used to ensure that no residual state is retained in the
    Keras session, which can help prevent memory leaks and ensure a clean
    state for subsequent operations.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    maneger_gpu.reset_keras()

def save_metrics(k,view,metrics, csv_filename):
    # Verifica se o arquivo já existe para evitar reescrita do cabeçalho
    """
    Saves metrics for a given view in a CSV file.

    Parameters
    ----------
    k : int
        The value of k for the current experiment.
    view : str
        The name of the view for the current experiment.
    metrics : dict
        A dictionary containing the metrics to be saved, with keys
        'accuracy', 'precision', 'recall', 'fscore', and 'kappa'.
    csv_filename : str
        The name of the CSV file to save the metrics in.

    Returns
    -------
    None

    Notes
    -----
    If the file does not exist, a header row will be written with the
    column names. If the file does exist, the metrics will be appended as
    a new row.
    """
    file_exists = os.path.exists(csv_filename)

    # Create and open CSV file for writing
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(["k","view", "accuracy", "precision", "recall", "fscore", "kappa"])
        
        # Write the values to the CSV
        writer.writerow([k, view, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['fscore'], metrics['kappa']]) 
    
    print(f"Metrics for k={k}, view={view} saved successfully in {csv_filename}.")

def gen_views(params, model, categories, k, folder, path_save, nm_model, view):
    # Equatorial views
    
    """
    Generate classification report and confusion matrix for the given model
    and parameters.

    Parameters:
    - params (dict): A dictionary with the following keys:
        - 'path_model': Path to the model (Keras .h5 file).
        - 'path_full_labels': Path to the labels directory.
        - 'path_test': Path to the test directory.
        - 'save_dir': Directory to save the reports.
        - 'size': Image size (width and height).
    - model (keras.Model): Trained model.
    - categories (list): List of categories.
    - k (int): Number of k-fold.
    - folder (str): Folder to save the reports.
    - nm_model (str): Model name.
    - view (str): View name (EQUATORIAL or POLAR).

    Returns:
    - matrix_fig (matplotlib.pyplot.Figure): Confusion matrix figure.
    - boxplot_fig (matplotlib.pyplot.Figure): Boxplot figure.
    """

    size=params['size']
    input_size = (size, size)

    path_test=f"{params['path_test']}k{k}/"
    view_dir = f"{path_test}{view}/" 
    print(f"view_dir: {view_dir}")

    test_generator = reports.load_data_test(view_dir, input_size)
    y_true_mapped, y_pred, present_labels, df_correct, df_incorrect=reports.predict_data_generator(
        model, 
        categories, 
        test_generator, 
        verbose=2)
    class_report=reports.generate_classification_report(y_true_mapped, 
                                            categories, 
                                            y_pred, 
                                            present_labels)
    
    # Confusion matrix
    matrix_fig, cm =reports.generate_confusion_matrix(y_true_mapped, 
                                        categories, 
                                        y_pred, 
                                        present_labels,
                                        normalize=False)
    boxplot_fig=reports.plot_confidence_boxplot(df_correct)

    metrics = reports.calculate_metrics(y_true_mapped, y_pred)
            
    # Save metrics and reports
    save_dir=folder
    if save_dir:
        df_mat = pd.DataFrame(cm, index=categories, columns=categories)
        df_correct.to_csv(f'{save_dir}/{nm_model}_{view}_df_correct_k{k}.csv')
        df_incorrect.to_csv(f'{save_dir}/{nm_model}_{view}_df_incorrect_k{k}.csv')
        class_report.to_csv(f"{save_dir}/{nm_model}_{view}_class_reports_k{k}.csv")
        df_mat.to_csv(f'{save_dir}/{nm_model}_{view}_confusion_matrix_k{k}.csv')
        
        matrix_fig.savefig(f'{save_dir}/{nm_model}_{view}_confusion_matrix_k{k}.jpg')
        boxplot_fig.savefig(f'{save_dir}/{nm_model}_{view}_boxplot_k{k}.jpg')

        save_metrics(k,view, metrics, f"{path_save}/{nm_model}_metrics.csv")

        print(f"✅ Relatório salvo em: {save_dir}")

    
def run(params):
    """
    Main function to generate reports for the given model and parameters.

    Parameters:
    - params (dict): A dictionary with the following keys:
        - 'path_model': Path to the model (Keras .h5 file).
        - 'path_labels': Path to the labels directory.
        - 'path_test': Path to the test directory.
        - 'save_dir': Directory to save the reports.
        - 'size': Image size (width and height).
    """
    
    print(f"params: {params}")
    #listar arquivos
    categories = sorted(os.listdir(params['path_full_labels']))
    print(f"total categories: {len(categories)}, categories: {categories}") 

    #Create folders test
    path=params['path_test']
    name_test=path.split("/")[1]
    path_save=f"{params['save_dir']}{name_test}/"
    print(f"path_dst: {path_save}")
    os.makedirs(path_save, exist_ok=True) 

    for i in range(10):
        reset_environment()
        k=i+1
        
        #create folders
        folder=f"{path_save}k{k}"
        print(f"folder: {folder}")
        os.makedirs(folder, exist_ok=True) 

        # Load model
        print(f"\n[INFO] load model")
        path_model = params['path_model']
        nm_model= path_model.split("/")[-2]
        path = f"{path_model}{nm_model}_bestLoss_{k}.keras"
        print(f"\n[INFO] path_model: {path}\n")
        model=None
        model = models.load_model(path)
        model.summary()  # Print model summary

        # Reports views
        gen_views(params, model, categories, k, folder, path_save, nm_model, "EQUATORIAL")
        gen_views(params, model, categories, k, folder, path_save, nm_model, "POLAR")
    
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
    config_file = args.config if args.config else 'phase3/config_test_views_B23.yaml'
    params = load_config(config_file)

    #run(params)
    #python phase3/reports_test_views.py --config phase3/config_test_views_B23.yaml
    
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