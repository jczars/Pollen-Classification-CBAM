import argparse
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import yaml

# Importing modules and functions
from models import grad_cam_lib as cam
from models import sound_test_finalizado

def predict_run(image, model, CATEGORIES):
    """
    Realiza a previsão de uma imagem com o modelo e retorna as probabilidades para todas as classes.

    Args:
        image (numpy.ndarray): A imagem a ser classificada.
        model (keras.Model): O modelo treinado.
        CATEGORIES (list): Lista de categorias correspondente às classes do modelo.

    Returns:
        numpy.ndarray: As probabilidades preditas para todas as classes.
    """
    # Adicionar uma dimensão extra para simular um lote
    image = np.expand_dims(image, axis=0) 
    
    # Fazer a previsão
    probs = model.predict(image)  # Probabilidades para todas as classes
    
    # Obter o índice da classe com a maior probabilidade
    predicted_class = np.argmax(probs, axis=1)[0]
    predict_label = CATEGORIES[predicted_class]
    
    # Obter a probabilidade da classe prevista
    prob = probs[0][predicted_class]
    
    # Exibir a classe prevista e sua probabilidade
    print(f"Index: {predicted_class}, Classe prevista: {predict_label}, Probabilidade: {prob}")
    
    return probs, predict_label

def display_cam_grid_vert(images, classes, model, conv_layer_name, CATEGORIES, n_cols=6, nrows=6):    
    fig, axs = plt.subplots(ncols=n_cols, nrows=nrows, figsize=(30, 30))
    
    for i in range(n_cols):
        
        img = images[i]
        
        probs, predict_label=predict_run(img, model, CATEGORIES)
        prob_values = probs[0]  # Presume-se que probs[0] contém os valores das probabilidades
            
        # original
        axs[0, i].imshow(img)
        axs[0, i].set_title("true: " + classes[i])
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        if i == 0:
            axs[0, i].set_ylabel(classes[i], rotation=0, ha="right")
    
        # Exibir probabilidades
        axs[1, i].barh(range(len(prob_values)), prob_values, color='b')
        axs[1, i].set_title("Probabilidades")
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks(range(len(CATEGORIES)))
        axs[1, i].set_yticklabels(CATEGORIES, fontsize=12)  # Reduz o tamanho da fonte
        axs[1, i].tick_params(axis='y', labelsize=8)       # Ajusta o tamanho dos rótulos
        axs[1, i].set_ylim(-0.5, len(CATEGORIES) - 0.5)    # Adiciona margens
        if i == 0:
            axs[1, i].set_ylabel("Scores", rotation=0, ha="right")
        
        # Grad-CAM
        heatmap=cam.make_gradcam_heatmap(img, model, conv_layer_name, pred_index=None)
        img_grad_cam =cam.superimpose_heatmap_on_image_0(img, heatmap, alpha=0.3)
        
        axs[2, i].imshow(img_grad_cam)
        axs[2, i].set_title("pred: " + predict_label)
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])
        if i == 0:
            axs[2, i].set_ylabel("Grad-CAM", rotation=0, ha="right")
        
        # Grad-CAM++
        heatmap_pp = cam.grad_cam_plus(model, img, conv_layer_name)
        img_grad_cam_plus_plus = cam.superimpose_heatmap_on_image_0(img, heatmap_pp, alpha=0.3)
        axs[3, i].imshow(img_grad_cam_plus_plus)
        #axs[3, i].set_title("pred: " + predict[i])
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])
        if i == 0:
            axs[3, i].set_ylabel("Grad-CAM++", rotation=0, ha="right")
    
        # Score-CAM
        heatmap_psc = cam.ScoreCam(model, img, conv_layer_name)
        img_score_cam = cam.superimpose_heatmap_on_image_0(img, heatmap_psc, alpha=0.3)
        axs[4, i].imshow(img_score_cam)
        #axs[4, i].set_title("pred: " + predict[i])
        axs[4, i].set_xticks([])
        axs[4, i].set_yticks([])
        if i == 0:
            axs[4, i].set_ylabel("Score-CAM", rotation=0, ha="right")
    
        # max activation
        act = cam.act_max(model, img, conv_layer_name, filter_numbers=10)
    
        axs[5, i].imshow(act)
        #axs[5, i].set_title("pred: " + predict[i])
        axs[5, i].set_xticks([])
        axs[5, i].set_yticks([])
        if i == 0:
            axs[5, i].set_ylabel("Max Activation", rotation=0, ha="right")
    
    return fig

def display_cam_grid_hor(images, classes, model, conv_layer_name, CATEGORIES, n_cols=6, n_rows=6):    
    # Ajustando a grade para orientação horizontal
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(30, 30))  # Inverte o número de colunas e linhas
    
    # Garantir que axs seja um array 2D
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(len(images)):  # Iterar sobre o número de imagens
        
        img = images[i]
        
        # Calcular as probabilidades
        probs, predict_label = predict_run(img, model, CATEGORIES)
        prob_values = probs[0]  # Assume-se que probs[0] contém os valores das probabilidades
            
        # Exibir imagem original
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("true: " + classes[i])
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
    
    
        # Exibir probabilidades
        axs[i, 1].barh(range(len(prob_values)), prob_values, color='b')
        axs[i, 1].set_title("Probabilidades")
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks(range(len(CATEGORIES)))
        axs[i, 1].set_yticklabels(CATEGORIES, fontsize=8)  # Reduz o tamanho da fonte
        axs[i, 1].tick_params(axis='y', labelsize=8)       # Ajusta o tamanho dos rótulos
        axs[i, 1].set_ylim(-0.5, len(CATEGORIES) - 0.5)    # Adiciona margens
    
        
        # Grad-CAM
        heatmap = cam.make_gradcam_heatmap(img, model, conv_layer_name, pred_index=None)
        img_grad_cam = cam.superimpose_heatmap_on_image_0(img, heatmap, alpha=0.3)
        
        axs[i, 2].imshow(img_grad_cam)
        axs[i, 2].set_title("Grad-CAM")
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])
        
        if i == 0:
            axs[i, 2].set_ylabel("predict; "+predict_label, rotation=0, ha="right")
        
        
        # Grad-CAM++
        heatmap_pp = cam.grad_cam_plus(model, img, conv_layer_name)
        img_grad_cam_plus_plus = cam.superimpose_heatmap_on_image_0(img, heatmap_pp, alpha=0.3)
        axs[i, 3].imshow(img_grad_cam_plus_plus)
        axs[i, 3].set_title("Grad-CAM++")
        axs[i, 3].set_xticks([])
        axs[i, 3].set_yticks([])
    
        # Score-CAM
        heatmap_psc = cam.ScoreCam(model, img, conv_layer_name)
        img_score_cam = cam.superimpose_heatmap_on_image_0(img, heatmap_psc, alpha=0.3)
        axs[i, 4].imshow(img_score_cam)
        axs[i, 4].set_title("Score-CAM")
        axs[i, 4].set_xticks([])
        axs[i, 4].set_yticks([])
    
        # Max Activation
        act = cam.act_max(model, img, conv_layer_name, filter_numbers=10)
        axs[i, 5].imshow(act)
        axs[i, 5].set_title("Max Activation")
        axs[i, 5].set_xticks([])
        axs[i, 5].set_yticks([])
    
    plt.tight_layout()
    return fig

import matplotlib.pyplot as plt
import numpy as np

def display_cam_grid1(images, classes, model, conv_layer_name, CATEGORIES):    
    # Determinar número de imagens
    n_images = len(images)
    assert 1 <= n_images <= 6, "O número de imagens deve estar entre 1 e 6."
    
    # Configurar dimensões do gráfico
    fig, axs = plt.subplots(nrows=n_images, ncols=6, figsize=(6 * 6, 5 * n_images))  # Ajuste o tamanho conforme necessário

    # Garantir que axs seja um array 2D
    if n_images == 1:
        axs = np.expand_dims(axs, axis=0)  # Assegurar que axs seja 2D
    
    for i in range(n_images):  # Iterar sobre o número de imagens
        img = images[i]
        
        # Calcular as probabilidades
        probs, predict_label = predict_run(img, model, CATEGORIES)
        prob_values = probs[0]  # Assume-se que probs[0] contém os valores das probabilidades
            
        # Exibir imagem original
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("True: " + classes[i], fontsize=12)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
    
        # Exibir probabilidades
        axs[i, 1].barh(range(len(prob_values)), prob_values, color='b')
        axs[i, 1].set_title(f"Predict: {predict_label}", fontsize=14)  # Aumentar o tamanho da fonte
        axs[i, 1].set_xticks(np.linspace(0, 1, 6))  # Definir as divisões para a escala de probabilidades
        axs[i, 1].set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)], fontsize=10)  # Ajustar rótulos
        axs[i, 1].set_yticks(range(len(CATEGORIES)))
        axs[i, 1].set_yticklabels(CATEGORIES, fontsize=12)  # Aumentar o tamanho da fonte
        axs[i, 1].tick_params(axis='y', labelsize=10)       # Ajustar o tamanho dos rótulos
        axs[i, 1].set_ylim(-0.5, len(CATEGORIES) - 0.5)    # Adicionar margens
        
        # Definir a escala de valores para as probabilidades
        axs[i, 1].set_xlim(0, 1)  # Adicionar limite para a escala de probabilidades
        axs[i, 1].tick_params(axis='x', labelsize=10)  # Ajuste do tamanho da fonte no eixo X
        axs[i, 1].set_xlabel('Probability', fontsize=12)  # Rótulo para a escala de probabilidades
    
        # Grad-CAM
        heatmap = cam.make_gradcam_heatmap(img, model, conv_layer_name, pred_index=None)
        img_grad_cam = cam.superimpose_heatmap_on_image_0(img, heatmap, alpha=0.3)
        axs[i, 2].imshow(img_grad_cam)
        axs[i, 2].set_title("Grad-CAM", fontsize=12)
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])
        
        # Grad-CAM++
        heatmap_pp = cam.grad_cam_plus(model, img, conv_layer_name)
        img_grad_cam_plus_plus = cam.superimpose_heatmap_on_image_0(img, heatmap_pp, alpha=0.3)
        axs[i, 3].imshow(img_grad_cam_plus_plus)
        axs[i, 3].set_title("Grad-CAM++", fontsize=12)
        axs[i, 3].set_xticks([])
        axs[i, 3].set_yticks([])
    
        # Score-CAM
        heatmap_psc = cam.ScoreCam(model, img, conv_layer_name)
        img_score_cam = cam.superimpose_heatmap_on_image_0(img, heatmap_psc, alpha=0.3)
        axs[i, 4].imshow(img_score_cam)
        axs[i, 4].set_title("Score-CAM", fontsize=12)
        axs[i, 4].set_xticks([])
        axs[i, 4].set_yticks([])
    
        # Max Activation
        act = cam.act_max(model, img, conv_layer_name, filter_numbers=10)
        axs[i, 5].imshow(act)
        axs[i, 5].set_title("Max Activation", fontsize=12)
        axs[i, 5].set_xticks([])
        axs[i, 5].set_yticks([])
    
    # Ajustar layout com espaçamento mínimo
    fig.subplots_adjust(wspace=0.01, hspace=0.1)  # Reduzir mais o wspace
    plt.tight_layout(pad=0.5)  # Ajustar o pad para um ajuste fino
    return fig

def display_cam_grid(images, classes, model, conv_layer_name, CATEGORIES):    
    # Determinar número de imagens
    n_images = len(images)
    assert 1 <= n_images <= 5, "O número de imagens deve estar entre 1 e 6."
    
    # Configurar dimensões do gráfico
    fig, axs = plt.subplots(nrows=n_images, ncols=5, figsize=(15, 8))  # Aumente o tamanho total da figura

    # Garantir que axs seja um array 2D
    if n_images == 1:
        axs = np.expand_dims(axs, axis=0)  # Assegurar que axs seja 2D
    
    for i in range(n_images):  # Iterar sobre o número de imagens
        img = images[i]
        
        # Calcular as probabilidades
        probs, predict_label = predict_run(img, model, CATEGORIES)
        prob_values = probs[0]  # Assume-se que probs[0] contém os valores das probabilidades
            
        # Exibir imagem original
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("True: " + classes[i], fontsize=12)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
    
        # Exibir probabilidades
        axs[i, 1].barh(range(len(prob_values)), prob_values, color='b')
        axs[i, 1].set_title(f"Predict: {predict_label}", fontsize=14)  # Aumentar o tamanho da fonte
        axs[i, 1].set_xticks(np.linspace(0, 1, 6))  # Definir as divisões para a escala de probabilidades
        axs[i, 1].set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)], fontsize=10)  # Ajustar rótulos
        axs[i, 1].set_yticks(range(len(CATEGORIES)))
        axs[i, 1].set_yticklabels(CATEGORIES, fontsize=8)  # Aumentar o tamanho da fonte
        axs[i, 1].tick_params(axis='y', labelsize=8)       # Ajustar o tamanho dos rótulos
        axs[i, 1].set_ylim(-0.5, len(CATEGORIES) - 0.5)    # Adicionar margens
        
        # Definir a escala de valores para as probabilidades
        axs[i, 1].set_xlim(0, 1)  # Adicionar limite para a escala de probabilidades
        axs[i, 1].tick_params(axis='x', labelsize=10)  # Ajuste do tamanho da fonte no eixo X
        #axs[i, 1].set_xlabel('Probability', fontsize=12)  # Rótulo para a escala de probabilidades
    
        # Grad-CAM
        heatmap = cam.make_gradcam_heatmap(img, model, conv_layer_name, pred_index=None)
        img_grad_cam = cam.superimpose_heatmap_on_image_0(img, heatmap, alpha=0.3)
        axs[i, 2].imshow(img_grad_cam)
        axs[i, 2].set_title("Grad-CAM", fontsize=12)
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])
        
        # Grad-CAM++
        heatmap_pp = cam.grad_cam_plus(model, img, conv_layer_name)
        img_grad_cam_plus_plus = cam.superimpose_heatmap_on_image_0(img, heatmap_pp, alpha=0.3)
        axs[i, 3].imshow(img_grad_cam_plus_plus)
        axs[i, 3].set_title("Grad-CAM++", fontsize=12)
        axs[i, 3].set_xticks([])
        axs[i, 3].set_yticks([])
    
        # Score-CAM
        heatmap_psc = cam.ScoreCam(model, img, conv_layer_name)
        img_score_cam = cam.superimpose_heatmap_on_image_0(img, heatmap_psc, alpha=0.3)
        axs[i, 4].imshow(img_score_cam)
        axs[i, 4].set_title("Score-CAM", fontsize=12)
        axs[i, 4].set_xticks([])
        axs[i, 4].set_yticks([])
    
    # Ajustar layout com espaçamento mínimo
    fig.subplots_adjust(wspace=0.1, hspace=0.2)  # Ajuste fino do espaçamento
    plt.tight_layout(pad=1)  # Ajustar o pad para margens controladas
    return fig

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


    model=tf.keras.models.load_model(path_model)
    CATEGORIES = sorted(os.listdir(path_data))
    images=load_imgs(path_data, images_list, target_size)

    fig=display_cam_grid(images, classes_list, model, conv_layer_name, CATEGORIES)
    
    saved_dir = "./interpretation/grad_cam/"
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(saved_dir), exist_ok=True)

    image_saved = os.path.join(saved_dir, nome)
    fig.savefig(image_saved)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data augmentation with specified configuration.")
    parser.add_argument("--config", type=str, default="./interpretation/config_class_well.yaml", 
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load parameters from config file and process augmentation
    #python3 preprocess/aug_balanc_bd_k.py --config preprocess/config_balanced.yaml
    config = load_config(args.config)
    run(config)
    sound_test_finalizado.beep(2)