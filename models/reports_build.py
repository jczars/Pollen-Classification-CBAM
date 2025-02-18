from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, cohen_kappa_score
import seaborn as sns


def predict_data_generator(test_data_generator, model, categories, batch_size, verbose=2):
    """
    Generates predictions and evaluation metrics (accuracy, precision, recall, fscore, kappa) for test data.
    Returns two DataFrames: one for correct predictions and one for incorrect predictions.

    Parameters:
        test_data_generator (ImageDataGenerator): Image Data Generator containing test data.
        model (keras.Model): Trained model used for prediction.
        categories (list): List of image class names.
        batch_size (int): Number of samples per batch.
        verbose (int): Verbosity level (0, 1, or 2). Default is 2.

    Returns:
        tuple: y_true (true labels), y_pred (predicted labels), 
               df_correct (DataFrame containing correctly classified samples), 
               df_incorrect (DataFrame containing incorrectly classified samples).
    """
    filenames = test_data_generator.filenames
    y_true = test_data_generator.classes
    df = pd.DataFrame(filenames, columns=['file'])
    confidences = []
    nb_samples = len(filenames)
    
    y_preds = model.predict(test_data_generator, steps=nb_samples // batch_size + 1)
    
    for prediction in y_preds:
        confidence = np.max(prediction)
        confidences.append(confidence)
        if verbose == 1:
            print(f'Prediction: {prediction}, Confidence: {confidence}')
    
    y_pred = np.argmax(y_preds, axis=1)
    
    if verbose == 2:
        print(f'Size y_true: {len(y_true)}')
        print(f'Size y_pred: {len(y_pred)}')
    
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    df['confidence'] = confidences
    df['true_label'] = [categories[i] for i in y_true]
    df['predicted_label'] = [categories[i] for i in y_pred]
    df['status'] = df.apply(lambda row: 'Correct' if row['y_true'] == row['y_pred'] else 'Incorrect', axis=1)

    # Separate the DataFrame into correct and incorrect predictions
    df_correct = df[df['status'] == 'Correct']
    df_incorrect = df[df['status'] == 'Incorrect']
    
    return y_true, y_pred, df_correct, df_incorrect

def plot_confusion_matrix(y_true, y_pred, categories, normalize=False):
    """
    Plots a confusion matrix for classification results, highlighting non-diagonal elements.

    Parameters:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        categories (list): List of class names.
        normalize (bool): Whether to normalize the confusion matrix values.

    Returns:
        tuple: fig (Matplotlib figure object of the confusion matrix) and mat (Confusion matrix as a NumPy array).
    """
    # Compute confusion matrix
    mat = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix if specified
    if normalize:
        mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
    
    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
    
    # Plot the confusion matrix using seaborn
    #sns.set(font_scale=0.8)
    sns.heatmap(mat, cmap="Blues", annot=False, 
                xticklabels=categories, yticklabels=categories, cbar=True, ax=ax, linewidths=0.5)

    # Set axis labels and title
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    # Rotate the tick labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=10)

    return fig, mat


def plot_confusion_matrixV3(y_true, y_pred, categories, normalize=False):
    """
    Plots a confusion matrix for classification results, highlighting non-diagonal elements.

    Parameters:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        categories (list): List of class names.
        normalize (bool): Whether to normalize the confusion matrix values.

    Returns:
        tuple: fig (Matplotlib figure object of the confusion matrix) and mat (Confusion matrix as a NumPy array).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np

    # Compute confusion matrix
    mat = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix if specified
    if normalize:
        mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
    
    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
    
    # Plot the confusion matrix using seaborn
    sns.set(font_scale=0.8)
    sns.heatmap(mat, cmap="Blues", annot=False, 
                xticklabels=categories, yticklabels=categories, cbar=True, ax=ax, linewidths=0.5)

    # Set axis labels and title
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    # Rotate the tick labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=10)

    # Annotate cells with the data
    for i in range(len(mat)):
        for j in range(len(mat)):
            value = mat[i, j]
            # Determine text color
            text_color = 'white' if i == j else 'black'
            # Highlight non-diagonal elements > 0 with a lighter red color
            bg_color = 'lightcoral' if i != j and value > 0 else None
            if bg_color:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=bg_color, alpha=0.5))
            ax.text(j + 0.5, i + 0.5, f'{value:.2f}' if normalize else f'{value}', 
                    ha='center', va='center', color=text_color, fontsize=10)

    plt.tight_layout()
    return fig, mat
def plot_confusion_matrixV4(y_true, y_pred, categories, normalize=False):
    """
    Plots a confusion matrix for classification results, highlighting non-diagonal elements.

    Parameters:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        categories (list): List of class names.
        normalize (bool): Whether to normalize the confusion matrix values.

    Returns:
        tuple: fig (Matplotlib figure object of the confusion matrix) and mat (Confusion matrix as a NumPy array).
        
    """
    

    # Compute confusion matrix
    mat = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix if specified
    if normalize:
        mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
    
    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
    
    # Plot the confusion matrix using seaborn
    sns.set(font_scale=0.8)
    sns.heatmap(mat, cmap="Blues", annot=True, 
                xticklabels=categories, yticklabels=categories, cbar=True, ax=ax, linewidths=0.5)

    # Set axis labels and title
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    # Rotate the tick labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=10)

    # Annotate non-diagonal cells with custom background and text color
    for i in range(len(mat)):
        for j in range(len(mat)):
            value = mat[i, j]
            if i != j:  # For off-diagonal elements
                # Highlight non-diagonal elements > 0 with a lighter red color
                if value > 0:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='lightcoral', alpha=0.5))
                # Add text with black color for off-diagonal cells
                ax.text(j + 0.5, i + 0.5, f'{value:.2f}' if normalize else f'{value}', 
                        ha='center', va='center', color='black', fontsize=10)

    plt.tight_layout()
    return fig, mat



def predict_unlabeled_data(test_data_generator, model, batch_size, categories, verbose=2):
    """
    Generates predictions and confidence scores for unlabeled data.

    Parameters:
        test_data_generator (ImageDataGenerator): Image Data Generator containing unlabeled data.
        model (keras.Model): Trained model for making predictions.
        batch_size (int): Number of samples per batch.
        categories (list): List of image class names.
        verbose (int): Verbosity level (0, 1, or 2). Default is 2.

    Returns:
        pd.DataFrame: DataFrame with predicted labels and confidence scores for each sample.
    """
    filenames = test_data_generator.filenames
    df = pd.DataFrame(filenames, columns=['file'])
    confidences = []
    nb_samples = len(filenames)
    print('Predicting unlabeled data...', nb_samples)
        
    y_preds = model.predict(test_data_generator, steps=nb_samples // batch_size + 1)
    
    for prediction in y_preds:
        confidence = np.max(prediction)
        confidences.append(confidence)
        if verbose == 1:
            print(f'Prediction: {prediction}, Confidence: {confidence}')
    
    predicted_labels = [categories[np.argmax(pred)] for pred in y_preds]
    
    df['labels'] = predicted_labels
    df['confidence'] = confidences
    
    return df

def calculate_metrics(y_true, y_pred):
    """
    Calculates and returns evaluation metrics: precision, recall, fscore, and kappa score.

    Parameters:
        y_true (list): True labels.
        y_pred (list): Predicted labels.

    Returns:
        dict: Dictionary containing precision, recall, fscore, and kappa score.
    """
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    metrics_dict = {
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'fscore': round(fscore, 3),
        'kappa': round(kappa, 3)
    }
    
    return metrics_dict

def generate_classification_report(y_true, y_pred, categories):
    """
    Generates a classification report.

    Parameters:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        categories (list): List of image class names.
        verbose (int): If 1, prints the classification report; if 0, does not print.

    Returns:
        pd.DataFrame: DataFrame with the classification report for each class.
    """
    report = classification_report(y_true, y_pred, target_names=categories, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    return df_report

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


def plot_training_metrics(history):
    """
    Plots training metrics (loss, accuracy) from the training history.

    Parameters:
        history (History): The training history returned by the Keras fit method.

    Returns:
        fig: Matplotlib figure object of the training metrics.
    """
    my_dpi = 100
    fig = plt.figure(figsize=(900 / my_dpi, 900 / my_dpi), dpi=my_dpi)
    pd_history = pd.DataFrame(history.history)
    pd_history.plot()
    plt.grid(True)

    return fig

def plot_training_metricsV1(history):
    """
    Plots training metrics (loss, accuracy) from the training history.

    Parameters:
        history (History): The training history returned by the Keras fit method.

    Returns:
        fig: Matplotlib figure object of the training metrics.
    """
    # Verifica se as chaves de métricas existem no histórico
    if not all(metric in history.history for metric in ['loss', 'accuracy']):
        raise KeyError("The training history is missing required metrics ('loss', 'accuracy').")
    
    # Converte o histórico para um DataFrame
    pd_history = pd.DataFrame(history.history)

    # Define o tamanho da figura
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 linha, 2 colunas de gráficos

    # Plotando o gráfico de perda
    axes[0].plot(pd_history['loss'], label='Loss', color='r', linestyle='-', linewidth=2)
    if 'val_loss' in pd_history.columns:
        axes[0].plot(pd_history['val_loss'], label='Validation Loss', color='orange', linestyle='--', linewidth=2)
    axes[0].set_title("Training Loss", fontsize=16)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    
    # Plotando o gráfico de acurácia
    axes[1].plot(pd_history['accuracy'], label='Accuracy', color='b', linestyle='-', linewidth=2)
    if 'val_accuracy' in pd_history.columns:
        axes[1].plot(pd_history['val_accuracy'], label='Validation Accuracy', color='green', linestyle='--', linewidth=2)
    axes[1].set_title("Training Accuracy", fontsize=16)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(loc='lower right')
    axes[1].grid(True)

    # Ajuste do layout
    plt.tight_layout()

    return fig

import matplotlib.pyplot as plt
import pandas as pd

def plot_training_metricsV2(history):
    """
    Plots training metrics (loss, accuracy) from the training history.

    Parameters:
        history (History): The training history returned by the Keras fit method.

    Returns:
        fig: Matplotlib figure object of the training metrics.
    """
    # Verifica se as chaves de métricas existem no histórico
    if not all(metric in history.history for metric in ['loss', 'accuracy']):
        raise KeyError("The training history is missing required metrics ('loss', 'accuracy').")
    
    # Converte o histórico para um DataFrame
    pd_history = pd.DataFrame(history.history)

    # Define o tamanho da figura
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 linha, 2 colunas de gráficos

    # Gráfico de Perda (Loss)
    axes[0].plot(pd_history['loss'], label='Loss', color='r', linestyle='-', linewidth=2)
    if 'val_loss' in pd_history.columns:
        axes[0].plot(pd_history['val_loss'], label='Validation Loss', color='orange', linestyle='--', linewidth=2)
    
    # Encontrar o menor valor de loss e val_loss
    best_loss_idx = pd_history['loss'].idxmin()
    best_loss = pd_history['loss'].min()
    
    axes[0].plot(best_loss_idx, best_loss, 'bo', label=f'Best Loss: {best_loss:.4f}')
    axes[0].annotate(f'{best_loss:.4f}', xy=(best_loss_idx, best_loss), 
                     xytext=(best_loss_idx, best_loss + 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

    if 'val_loss' in pd_history.columns:
        best_val_loss_idx = pd_history['val_loss'].idxmin()
        best_val_loss = pd_history['val_loss'].min()
        
        axes[0].plot(best_val_loss_idx, best_val_loss, 'go', label=f'Best Val Loss: {best_val_loss:.4f}')
        axes[0].annotate(f'{best_val_loss:.4f}', xy=(best_val_loss_idx, best_val_loss), 
                         xytext=(best_val_loss_idx, best_val_loss + 0.1),
                         arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    axes[0].set_title("Training Loss", fontsize=16)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Gráfico de Acurácia (Accuracy)
    axes[1].plot(pd_history['accuracy'], label='Accuracy', color='b', linestyle='-', linewidth=2)
    if 'val_accuracy' in pd_history.columns:
        axes[1].plot(pd_history['val_accuracy'], label='Validation Accuracy', color='green', linestyle='--', linewidth=2)
    
    # Encontrar o melhor valor de acurácia (opcional)
    best_acc_idx = pd_history['accuracy'].idxmax()
    best_acc = pd_history['accuracy'].max()
    axes[1].plot(best_acc_idx, best_acc, 'bo', label=f'Best Acc: {best_acc:.4f}')
    
    if 'val_accuracy' in pd_history.columns:
        best_val_acc_idx = pd_history['val_accuracy'].idxmax()
        best_val_acc = pd_history['val_accuracy'].max()
        axes[1].plot(best_val_acc_idx, best_val_acc, 'go', label=f'Best Val Acc: {best_val_acc:.4f}')
    
    axes[1].set_title("Training Accuracy", fontsize=16)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(loc='lower right')
    axes[1].grid(True)

    # Ajuste do layout
    plt.tight_layout()

    return fig

import matplotlib.pyplot as plt
import pandas as pd

def plot_training_metricsV3(history, reduce_lr_on_plateau_epochs=None):
    """
    Plots training metrics (loss and accuracy) from the training history.

    Parameters:
        history (History): The training history returned by the Keras fit method.
        reduce_lr_on_plateau_epochs (list or None): List of epochs where ReduceLROnPlateau was triggered.

    Returns:
        fig: Matplotlib figure object of the training metrics.
    """
    # Verifica se as chaves de métricas existem no histórico
    if not all(metric in history.history for metric in ['loss', 'accuracy']):
        raise KeyError("The training history is missing required metrics ('loss', 'accuracy').")
    
    # Converte o histórico para um DataFrame
    pd_history = pd.DataFrame(history.history)

    # Define o tamanho da figura
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 linha, 2 colunas de gráficos

    # Gráfico de Perda (Loss)
    axes[0].plot(pd_history['loss'], label='Loss', color='r', linestyle='-', linewidth=2)
    if 'val_loss' in pd_history.columns:
        axes[0].plot(pd_history['val_loss'], label='Validation Loss', color='orange', linestyle='--', linewidth=2)
    
    # Identificar e destacar os pontos de ReduceLROnPlateau
    if reduce_lr_on_plateau_epochs:
        for epoch in reduce_lr_on_plateau_epochs:
            axes[0].axvline(x=epoch, color='purple', linestyle='--', alpha=0.7, label='ReduceLR Plateau')

    # Encontrar o menor valor de loss e val_loss
    best_loss_idx = pd_history['loss'].idxmin()
    best_loss = pd_history['loss'].min()
    
    axes[0].plot(best_loss_idx, best_loss, 'bo', label=f'Best Loss: {best_loss:.4f}')
    axes[0].annotate(f'{best_loss:.4f}', xy=(best_loss_idx, best_loss), 
                     xytext=(best_loss_idx, best_loss + 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

    if 'val_loss' in pd_history.columns:
        best_val_loss_idx = pd_history['val_loss'].idxmin()
        best_val_loss = pd_history['val_loss'].min()
        
        axes[0].plot(best_val_loss_idx, best_val_loss, 'go', label=f'Best Val Loss: {best_val_loss:.4f}')
        axes[0].annotate(f'{best_val_loss:.4f}', xy=(best_val_loss_idx, best_val_loss), 
                         xytext=(best_val_loss_idx, best_val_loss + 0.1),
                         arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    axes[0].set_title("Training Loss", fontsize=16)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Gráfico de Acurácia (Accuracy)
    axes[1].plot(pd_history['accuracy'], label='Accuracy', color='b', linestyle='-', linewidth=2)
    if 'val_accuracy' in pd_history.columns:
        axes[1].plot(pd_history['val_accuracy'], label='Validation Accuracy', color='green', linestyle='--', linewidth=2)
    
    # Encontrar o melhor valor de acurácia
    best_acc_idx = pd_history['accuracy'].idxmax()
    best_acc = pd_history['accuracy'].max()
    axes[1].plot(best_acc_idx, best_acc, 'bo', label=f'Best Acc: {best_acc:.4f}')
    
    if 'val_accuracy' in pd_history.columns:
        best_val_acc_idx = pd_history['val_accuracy'].idxmax()
        best_val_acc = pd_history['val_accuracy'].max()
        axes[1].plot(best_val_acc_idx, best_val_acc, 'go', label=f'Best Val Acc: {best_val_acc:.4f}')
    
    axes[1].set_title("Training Accuracy", fontsize=16)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(loc='lower right')
    axes[1].grid(True)

    # Ajuste do layout
    plt.tight_layout()

    return fig

def plot_training_metricsV4(history, reduce_lr_on_plateau_epochs=None):
    """
    Plots training metrics (loss, accuracy, and learning rate) from the training history.

    Parameters:
        history (History): The training history returned by the Keras fit method.
        reduce_lr_on_plateau_epochs (list or None): List of epochs where ReduceLROnPlateau was triggered.

    Returns:
        fig: Matplotlib figure object of the training metrics.
    """
    # Verifica se as chaves de métricas existem no histórico
    if not all(metric in history.history for metric in ['loss', 'accuracy']):
        raise KeyError("The training history is missing required metrics ('loss', 'accuracy').")
    
    # Converte o histórico para um DataFrame
    pd_history = pd.DataFrame(history.history)

    # Define o tamanho da figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 linha, 3 colunas de gráficos

    # Gráfico de Perda (Loss)
    axes[0].plot(pd_history['loss'], label='Loss', color='r', linestyle='-', linewidth=2)
    if 'val_loss' in pd_history.columns:
        axes[0].plot(pd_history['val_loss'], label='Validation Loss', color='orange', linestyle='--', linewidth=2)
    
    # Identificar e destacar os pontos de ReduceLROnPlateau
    if reduce_lr_on_plateau_epochs:
        for epoch in reduce_lr_on_plateau_epochs:
            axes[0].axvline(x=epoch, color='purple', linestyle='--', alpha=0.7, label='ReduceLR Plateau')

    # Encontrar o menor valor de loss e val_loss
    best_loss_idx = pd_history['loss'].idxmin()
    best_loss = pd_history['loss'].min()
    axes[0].plot(best_loss_idx, best_loss, 'bo', label=f'Best Loss: {best_loss:.4f}')
    axes[0].annotate(f'{best_loss:.4f}', xy=(best_loss_idx, best_loss), 
                     xytext=(best_loss_idx, best_loss + 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

    if 'val_loss' in pd_history.columns:
        best_val_loss_idx = pd_history['val_loss'].idxmin()
        best_val_loss = pd_history['val_loss'].min()
        axes[0].plot(best_val_loss_idx, best_val_loss, 'go', label=f'Best Val Loss: {best_val_loss:.4f}')
        axes[0].annotate(f'{best_val_loss:.4f}', xy=(best_val_loss_idx, best_val_loss), 
                         xytext=(best_val_loss_idx, best_val_loss + 0.1),
                         arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)
    
    axes[0].set_title("Training Loss", fontsize=16)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    # Gráfico de Acurácia (Accuracy)
    axes[1].plot(pd_history['accuracy'], label='Accuracy', color='b', linestyle='-', linewidth=2)
    if 'val_accuracy' in pd_history.columns:
        axes[1].plot(pd_history['val_accuracy'], label='Validation Accuracy', color='green', linestyle='--', linewidth=2)
    
    # Encontrar o melhor valor de acurácia
    best_acc_idx = pd_history['accuracy'].idxmax()
    best_acc = pd_history['accuracy'].max()
    axes[1].plot(best_acc_idx, best_acc, 'bo', label=f'Best Acc: {best_acc:.4f}')
    
    if 'val_accuracy' in pd_history.columns:
        best_val_acc_idx = pd_history['val_accuracy'].idxmax()
        best_val_acc = pd_history['val_accuracy'].max()
        axes[1].plot(best_val_acc_idx, best_val_acc, 'go', label=f'Best Val Acc: {best_val_acc:.4f}')
    
    axes[1].set_title("Training Accuracy", fontsize=16)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(loc='lower right')
    axes[1].grid(True)

    # Gráfico de Learning Rate
    if 'lr' in pd_history.columns:
        axes[2].plot(pd_history['lr'], label='Learning Rate', color='purple', linestyle='-', linewidth=2)
        axes[2].set_title("Learning Rate", fontsize=16)
        axes[2].set_xlabel('Epochs', fontsize=12)
        axes[2].set_ylabel('Learning Rate', fontsize=12)
        axes[2].legend(loc='upper right')
        axes[2].grid(True)
    else:
        axes[2].text(0.5, 0.5, 'No Learning Rate data', ha='center', va='center', fontsize=12, color='red')
    
    # Ajuste do layout
    plt.tight_layout()

    return fig




def reports_gen(test_data_generator, model, categories, history, reports_config):
    """
    Generate and save evaluation reports for the trained model.

    Parameters:
        test_data_generator (ImageDataGenerator): Data generator for the test dataset.
        model (keras.Model): Trained model to evaluate.
        categories (list): List of categories for the evaluation.
        history (History): History object returned from the training process.
        reports_config (dict): Configuration dictionary containing parameters for report generation.

    Returns:
        dict: A dictionary containing evaluation metrics such as test loss, test accuracy, precision, recall, f-score, and kappa.
    """
    save_dir = reports_config['save_dir']
    k = reports_config['time']
    batch_size = reports_config['batch_size']    
    id_test = reports_config['id_test']
    nm_model = f"{id_test}_{reports_config['model']}"

    # Evaluate the model
    (test_loss, test_accuracy) = model.evaluate(test_data_generator, batch_size=batch_size, verbose=1)
    
    # Predict and generate reports
    y_true, y_pred, df_correct, df_incorrect = predict_data_generator(
        test_data_generator,
        model,
        categories,
        batch_size,
        verbose=2
    )
    
    # Confusion matrix
    matrix_fig, mat = plot_confusion_matrixV4(y_true, y_pred, categories)
    df_mat = pd.DataFrame(mat, index=categories, columns=categories)
    
    # Boxplot, classification report, and training metrics
    boxplot_fig = plot_confidence_boxplot(df_correct)
    class_report = generate_classification_report(y_true, y_pred, categories)
    metrics = calculate_metrics(y_true, y_pred)
    figTrain = plot_training_metricsV3(history)

    metrics_all = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'fscore': metrics['fscore'],
        'kappa': metrics['kappa']   
    }
    
    # Save metrics and reports
    if save_dir:
        df_correct.to_csv(f'{save_dir}/Test_{nm_model}_df_correct_T{k}.csv')
        df_incorrect.to_csv(f'{save_dir}/Test_{nm_model}_df_incorrect_T{k}.csv')
        class_report.to_csv(f"{save_dir}/Test_{nm_model}_classif_report_T{k}.csv")
        df_mat.to_csv(f'{save_dir}//Test_{nm_model}_conf_matrix_T{k}.csv')

        metrics_df = pd.DataFrame([metrics_all])
        metrics_df.to_csv(f"{save_dir}/Test_{nm_model}_metrics_{nm_model}_T{k}.csv")

        figTrain.savefig(f'{save_dir}/Test_{nm_model}__TrainLoss_T{k}.jpg')
        matrix_fig.savefig(f'{save_dir}/Test_{nm_model}__conf_matrix_T{k}.jpg')
        boxplot_fig.savefig(f'{save_dir}/Test_{nm_model}_boxplot_T{k}.jpg')
    
    # Return results dictionary
    return metrics_all

if __name__ == "__main__":
    help(predict_data_generator)
    help(plot_confusion_matrix)
    help(predict_unlabeled_data)
    help(calculate_metrics)
    help(generate_classification_report)
    help(plot_confidence_boxplot)
