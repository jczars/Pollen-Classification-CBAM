import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, classification_report

def sum_and_plot_confusion_matrices(folder_path, output_csv_path, output_image_path, normalize=False):
    """
    Reads all confusion matrix CSV files with a specific naming pattern in a folder, sums them, 
    and plots the resulting matrix as an image.

    Parameters:
        folder_path (str): Path to the folder containing confusion matrix CSV files.
        output_csv_path (str): Path to save the summed confusion matrix as a CSV file.
        output_image_path (str): Path to save the plotted confusion matrix image.
        normalize (bool): Whether to normalize the confusion matrix values.

    Returns:
        pd.DataFrame: DataFrame of the summed confusion matrix.
    """

    test_id, model_name = extract_test_info(folder_path)
    print(f"test_id {test_id} model_name {model_name}")

    print(f"Test_{test_id}_{test_id}_{model_name}_mat_conf_k")

    # List all CSV files matching the naming pattern in the folder
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if f.startswith(f"Test_{test_id}_{test_id}_{model_name}_mat_conf_k") and f.endswith('.csv')]

    if not csv_files:
        print("No valid matrices found in the folder.")
        return

    # Initialize the accumulated sum
    matrix_sum = None

    # Loop through all matching CSV files and accumulate the matrices
    for file in csv_files:
        matrix = pd.read_csv(file).set_index('Unnamed: 0')
        
        if matrix_sum is None:
            # Initialize with the first matrix
            matrix_sum = matrix
        else:
            # Sum the current matrix with the accumulated sum
            matrix_sum += matrix

    # Save the summed matrix to a CSV file
    matrix_sum.to_csv(output_csv_path)
    print(f"The sum of all matrices has been saved to the file: {output_csv_path}")

    # Convert the summed matrix to a NumPy array and get class labels
    mat = matrix_sum.values
    categories = matrix_sum.index.tolist()

    # Normalize the matrix if required
    if normalize:
        mat = mat.astype('float') / mat.sum(axis=1, keepdims=True)
    
    # Choose the format dynamically based on normalization
    fmt = ".2f" if normalize else "d"

    # Plot the summed confusion matrix
    fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
    sns.set(font_scale=0.8)
    sns.heatmap(mat, cmap="Blues", annot=True, fmt=fmt,  # Force integer display
            xticklabels=categories, yticklabels=categories, cbar=True, ax=ax, linewidths=0.5)

    # Set axis labels and title
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_title('Consolidated confusion matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    # Rotate the tick labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=10)

    plt.tight_layout()
    fig.savefig(output_image_path)
    plt.close(fig)
    print(f"The resulting confusion matrix has been saved as an image in the file: {output_image_path}")

    return matrix_sum


def classification_report_from_conf_matrix(conf_matrix_df):
    """
    Generates a classification report from a confusion matrix loaded into a DataFrame.

    Parameters:
        conf_matrix_df (pd.DataFrame): DataFrame containing the confusion matrix.

    Returns:
        pd.DataFrame: DataFrame containing the detailed per-class report.
    """
    # Convert the confusion matrix into a NumPy array
    conf_matrix = conf_matrix_df.values
    labels = conf_matrix_df.index.tolist()

    # Rebuild the true and predicted labels
    y_true = np.concatenate([np.full(conf_matrix[i, j], labels[i]) for i in range(len(labels)) for j in range(len(labels))])
    y_pred = np.concatenate([np.full(conf_matrix[i, j], labels[j]) for i in range(len(labels)) for j in range(len(labels))])

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Generate the detailed per-class report as a dictionary
    class_report_dict = classification_report(y_true, y_pred, target_names=labels, zero_division=0, output_dict=True)

    # Convert the dictionary into a DataFrame
    class_report_df = pd.DataFrame(class_report_dict).transpose()

    # Round to 3 decimal places (except 'support', which should be an integer)
    for col in ['precision', 'recall', 'f1-score']:
        class_report_df[col] = class_report_df[col].round(3)
    
    # Ensure 'support' is an integer
    class_report_df['support'] = class_report_df['support'].astype(int)

    # Return the classification report DataFrame
    return class_report_df


def performance_report_pd0(arr, classes, save_dir, id_test, nm_model):
  mat_df=pd.DataFrame(columns= ['Classes','Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support'])
  cr = dict()
  # col=number of class
  col=len(arr)
  support_sum = 0
  for i in range(col):
    vertical_sum= sum([arr[j][i] for j in range(col)])
    horizontal_sum= sum(arr[i])
    #https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models
    a = round(arr[i][i] / horizontal_sum, 2)
    p = round(arr[i][i] / vertical_sum, 2)
    r = round(arr[i][i] / horizontal_sum, 2)
    f = round((2 * p * r) / (p + r), 2)
    s = horizontal_sum
    row=[classes[i],a,p,r,f,s]
    support_sum+=s
    cr[i]=row
    mat_df.loc[i]=row

    # Salva o relatório de classificação
    
    df_report = pd.DataFrame(mat_df)
    if not (save_dir == ''):
        df_report.to_csv(save_dir+'class_report_test_'+str(id_test)+'_'+nm_model+'.csv', index=True)
  return mat_df

def performance_report_from_df(conf_matrix_df, save_dir, folder_path):
    """
    Gera um relatório de métricas a partir de um DataFrame de matriz de confusão.

    Parâmetros:
        conf_matrix_df (pd.DataFrame): DataFrame contendo a matriz de confusão.
        save_dir (str): Diretório para salvar o relatório (opcional).
        id_test (str): Identificador do teste (opcional, usado no nome do arquivo).
        nm_model (str): Nome do modelo (opcional, usado no nome do arquivo).

    Retorna:
        pd.DataFrame: DataFrame com as métricas de classificação.
    """

    test_id, model_name = extract_test_info(folder_path)
    # Extrai a matriz de confusão (ignorando a coluna de nomes das classes)
    
    # Convert the confusion matrix into a NumPy array
    arr = conf_matrix_df.values
    classes = conf_matrix_df.index.tolist()

    # Inicializa o DataFrame para métricas
    metrics_df = pd.DataFrame(columns=['Classes', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Support'])

    # Calcula métricas para cada classe
    for i, class_name in enumerate(classes):
        vertical_sum = arr[:, i].sum()  # Soma vertical (preditos como classe i)
        horizontal_sum = arr[i].sum()  # Soma horizontal (verdadeiros da classe i)
        true_positive = arr[i, i]  # Verdadeiros positivos para a classe i

        # Cálculo de métricas com tratamento de divisão por zero
        accuracy = round(true_positive / horizontal_sum, 3) if horizontal_sum > 0 else 0.0
        precision = round(true_positive / vertical_sum, 3) if vertical_sum > 0 else 0.0
        recall = round(true_positive / horizontal_sum, 3) if horizontal_sum > 0 else 0.0
        f1_score = round((2 * precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0.0
        support = horizontal_sum  # Suporte

        # Adiciona os dados ao DataFrame
        metrics_df.loc[i] = [class_name, accuracy, precision, recall, f1_score, support]

    # Salva o relatório em CSV (se um diretório for fornecido)
    if save_dir:
        filename = f"{save_dir}/class_report_test_{test_id}_{model_name}.csv"
        metrics_df.to_csv(filename, sep=';', decimal=',', index=True)


    return metrics_df

def save_classification_report(class_report_df, output_csv_path):
    """
    Saves the classification report to a CSV file, using a comma as the decimal separator.

    Parameters:
        class_report_df (pd.DataFrame): DataFrame containing the classification report.
        output_csv_path (str): Path to the output CSV file.

    Returns:
        None
    """
    # Save the DataFrame with a comma as the decimal separator
    class_report_df.to_csv(output_csv_path, sep=';', decimal=',', index=True)
    print(f"Classification report saved to: {output_csv_path}")

def saved(folder):
    # Automate the output path
    save_dir = os.path.join(
        os.path.dirname(folder.rstrip('/')),  # Remove the last slash and get the parent directory
        os.path.basename(folder.rstrip('/')) + '_consolidated/'  # Add '_aggregated' to the folder name
    )

    # Create the output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    print(f"Automated save directory: {save_dir}")
    return save_dir  # Return the directory for future use

def extract_test_info(folder_path):
    """
    Extracts the test id and model name from the folder path.

    Parameters:
        folder_path (str): Path to the folder containing the test name.

    Returns:
        tuple: test id and model name as strings.
    """
    folder_name = os.path.basename(folder_path.rstrip('/'))  # Remove the final slash and get the folder name
    parts = folder_name.split('_', 1)  # Split the folder name at the first occurrence of "_"
    
    # Ensure the folder name is correctly structured and split
    if len(parts) >= 2:
        test_id = parts[0]  # The first part is the test_id
        model_name = parts[1].split('_')[0]  # The second part is the model name (before any subsequent '_')
        return test_id, model_name
    else:
        raise ValueError(f"The folder name '{folder_name}' does not match the expected format.")

def run(folder, normalize=False):
    save_dir = saved(folder)
    mat_csv = os.path.join(save_dir, 'consolidated_confusion_matrix.csv')
    mat_image = os.path.join(save_dir, 'consolidated_confusion_matrix.png')
    output_reports_csv = os.path.join(save_dir, 'consolidated_classification_report.csv')


    # Process confusion matrices and generate the report
    conf_matrix_df = sum_and_plot_confusion_matrices(folder, mat_csv, mat_image, normalize=normalize)
    if conf_matrix_df is not None:
        report_df = classification_report_from_conf_matrix(conf_matrix_df)
        save_classification_report(report_df, output_reports_csv)
        print(f"Classification report saved to: {output_reports_csv}")

        performance_report_from_df(conf_matrix_df, save_dir, folder)


if __name__ == "__main__":
    # Example usage:
    folder = './results/phase2/reports_cr_13_500/0_DenseNet201_reports/'
    normalize=False
    run(folder, normalize)
    

