import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def load_data_test(test_dir, input_size, batch_size=32):
    """
    Loads test data from a directory.

    Parameters
    ----------
    test_dir : str
        Path to the directory containing test data.
    input_size : tuple
        Size of input images.
    batch_size : int, optional
        Batch size for the data generator.

    Returns
    -------
    test_generator : ImageDataGenerator
        Data generator for test data.
    """
    idg = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = idg.flow_from_directory(
        test_dir,
        target_size=input_size,
        class_mode="categorical",
        shuffle=False,
        batch_size=batch_size
    )
    return test_generator

def predict_data_generator(model, categories, test_generator, verbose=2):
    """
    Generates predictions and evaluation metrics (accuracy, precision, recall, fscore, kappa) for test data.
    Returns two DataFrames: one for correct predictions and one for incorrect predictions.

    Parameters:
        model (keras.Model): Trained model used for prediction.
        categories (list): List of image class names.
        test_generator (ImageDataGenerator): Image Data Generator containing test data.
        save_dir (str): Directory to save the reports.
        verbose (int): Verbosity level (0, 1, or 2). Default is 2.

    Returns:
        tuple: y_true (true labels), y_pred (predicted labels), 
               df_correct (DataFrame containing correctly classified samples), 
               df_incorrect (DataFrame containing incorrectly classified samples).
    """
    filenames = test_generator.filenames
    y_true = test_generator.classes  
    df = pd.DataFrame(filenames, columns=['file'])
    confidences = [] 

    test_class_names = list(test_generator.class_indices.keys())  
    label_mapping = {test_class_names[i]: categories.index(test_class_names[i]) for i in range(len(test_class_names))}
    y_true_mapped = np.array([label_mapping[test_class_names[i]] for i in y_true])
    
    y_preds = model.predict(test_generator, verbose=1)

    for prediction in y_preds:
        confidence = np.max(prediction)
        confidences.append(confidence)
        if verbose == 1:
            print(f'Prediction: {prediction}, Confidence: {confidence}')
    
    y_pred = np.argmax(y_preds, axis=1)
    present_labels = sorted(set(y_true_mapped))  

    if verbose == 2:
        print(f'Size y_true: {len(y_true)}')
        print(f'Size y_pred: {len(y_pred)}')
    
    df['y_true'] = y_true_mapped
    df['y_pred'] = y_pred
    df['confidence'] = confidences
    df['true_label'] = [categories[i] for i in y_true]
    df['predicted_label'] = [categories[i] for i in y_pred]
    df['status'] = df.apply(lambda row: 'Correct' if row['y_true'] == row['y_pred'] else 'Incorrect', axis=1)

    # Separate the DataFrame into correct and incorrect predictions
    df_correct = df[df['status'] == 'Correct']    
    df_incorrect = df[df['status'] == 'Incorrect']

    return y_true_mapped, y_pred, present_labels, df_correct, df_incorrect



def generate_classification_report(y_true_mapped, categories, y_pred, present_labels):
    """
    Generates a classification report and saves it as a CSV file.

    Parameters:
        y_true_mapped (array-like): True labels mapped to indices.
        categories (list): List of category names.
        y_pred (array-like): Predicted labels.
        present_labels (list): List of labels present in the predictions.
        output_csv (str, optional): File path to save the CSV report. Defaults to "relatorio.csv".

    Returns:
        None
    """

    report = classification_report(y_true_mapped, y_pred, labels=present_labels, 
                                   target_names=[categories[i] for i in present_labels], 
                                   output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()

    return df_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_confusion_matrix(y_true_mapped, categories, y_pred, present_labels, normalize=False):
    """
    Generates and displays a confusion matrix, ensuring alignment between true categories 
    and predicted labels. Highlights missing classes and misclassified cases.

    Parameters:
        y_true_mapped (array-like): True labels mapped to indices.
        categories (list): List of all class names.
        y_pred (array-like): Predicted labels.
        present_labels (list): Sorted list of labels present in the test set.
        normalize (bool, optional): Whether to normalize the confusion matrix values. Defaults to False.

    Returns:
        fig (matplotlib.figure.Figure): The generated confusion matrix figure.
        df_cm (pandas.DataFrame): Confusion matrix as a DataFrame.
    """

    num_classes = len(categories)  # Total number of possible classes
    absent_labels = sorted(set(range(num_classes)) - set(present_labels))

    # Reduced confusion matrix (only for present classes)
    cm_reduced = confusion_matrix(y_true_mapped, y_pred, labels=present_labels)

    # Normalize if required
    if normalize:
        cm_reduced = np.divide(cm_reduced.astype('float'), cm_reduced.sum(axis=1, keepdims=True), 
                               where=cm_reduced.sum(axis=1, keepdims=True) != 0)

    # Create full-size confusion matrix (num_classes x num_classes)
    full_cm = np.zeros((num_classes, num_classes))
    for i, real in enumerate(present_labels):
        for j, pred in enumerate(present_labels):
            full_cm[real, pred] = cm_reduced[i, j]

    # Create a DataFrame correctly aligned
    df_cm = pd.DataFrame(full_cm, index=categories, columns=categories)

    # Generate heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.color_palette("Blues", as_cmap=True)  

    mat = full_cm  # Alias for readability
    sns.heatmap(mat, annot=True, fmt=".2f" if normalize else "g", cmap=cmap, 
                xticklabels=categories, yticklabels=categories, cbar=True, linewidths=0.5, ax=ax)

    # Highlight missing labels and adjust rotation
    for i in absent_labels:
        ax.get_yticklabels()[i].set_backgroundcolor("yellow")  
        ax.get_xticklabels()[i].set_backgroundcolor("yellow")  
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation=90)  # Fixed rotation
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)

    # Annotate non-diagonal cells with custom background and text color
    for i in range(len(mat)):
        for j in range(len(mat)):
            value = mat[i, j]
            if i != j:  # Only for off-diagonal elements
                if value > 0:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='lightcoral', alpha=0.5))
                ax.text(j + 0.5, i + 0.5, f'{value:.2f}' if normalize else f'{int(value)}', 
                        ha='center', va='center', color='black', fontsize=10)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else "") + "\n(Yellow = Missing Classes)")

    plt.tight_layout()
    return fig, df_cm



def plot_confidence_boxplot(df_correct):
    """
    Creates a boxplot of confidence scores for correctly classified samples.

    Parameters:
        df_correct (pd.DataFrame): DataFrame containing correctly classified samples.

    Returns:
        fig: Matplotlib figure object of the boxplot.
    """
    # Set up the figure and its dimensions
    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    
    # Customize the plot style
    sns.set_style("whitegrid")
    
    # Plot the boxplot using seaborn
    #sns.boxplot(data=df_correct, y="true_label", x="confidence", ax=ax, palette="Blues")
    sns.boxplot(data=df_correct, x="confidence", y="true_label", ax=ax, hue="true_label", 
                palette="Blues", showfliers=True)
    
    # Set up the plot title and labels
    ax.set_title("Confidence Scores for Correct Classifications", fontsize=16)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    
    # Improve readability of y-axis labels
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10, rotation=0)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    help(load_data_test)
    help(predict_data_generator)
    help(generate_classification_report)
    help(generate_confusion_matrix)
