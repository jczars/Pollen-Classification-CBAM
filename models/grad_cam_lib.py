#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:19:09 2024

@author: jczars
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
#from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import copy
import cv2
import os


def load_model(path_model):
    model_rec=tf.keras.models.load_model(path_model)
    model_rec.summary()

    return model_rec

def load_img_gen(img_path, target_size, verbose=0):
    # Carregar imagem com tamanho alvo
    img = load_img(img_path, target_size=target_size)
    
    # Converter a imagem em um array numpy
    img_array = img_to_array(img)
    
    # Adicionar uma dimensão extra para simular um lote (batch)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Criar o ImageDataGenerator para pré-processamento
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Aplicar o ImageDataGenerator na imagem carregada
    img_iterator = datagen.flow(img_array)
    
    # Acessar a imagem processada
    processed_image = img_iterator.next()
    
    # Visualizar a imagem processada (opcional)
    if verbose==1:
        plt.imshow(processed_image[0])
        plt.show()

    return processed_image[0]

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

def grad_cam_plus(model, img,
                  layer_name="block5_conv3", label_name=None,
                  category_id=None):
    """Get a heatmap by Grad-CAM++.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)

    #conv_layer = model.get_layer(layer_name)
    #heatmap_model = Model([model.inputs], [conv_layer.output, model.output])
    heatmap_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id is None:
                    category_id = np.argmax(predictions[0])
                if label_name is not None:
                    print(label_name[category_id])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0,1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
    grad_cam_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    heatmap = np.maximum(grad_cam_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap

def ScoreCam(model, img_array, layer_name, max_N=-1):
    # Adicionar uma dimensão extra para simular um lote
    img_array = np.expand_dims(img_array, axis=0) 
    
    cls = np.argmax(model.predict(img_array))
    act_map_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
    
    # extract effective maps
    if max_N != -1:
        act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
        unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
        max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
        act_map_array = act_map_array[:,:,:,max_N_indices]

    input_shape = model.layers[0].output_shape[0][1:]  # get input shape
    # 1. upsample to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0,:,:,k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)
    # 4. feed masked inputs into CNN model and softmax
    pred_from_masked_input_array = softmax(model.predict(masked_input_array))
    # 5. define weight as the score of target class
    weights = pred_from_masked_input_array[:,cls]
    # 6. get final class discriminative localization map as linear weighted combination of all activation maps
    cam = np.dot(act_map_array[0,:,:,:], weights)
    cam = np.maximum(0, cam)  # Passing through ReLU
    cam /= np.max(cam)  # scale 0 to 1.0
    
    return cam

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Adicionar uma dimensão extra para simular um lote
    img_array = np.expand_dims(img_array, axis=0) 
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy() 

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
    return f

def superimpose_heatmap_on_image(img_src, heatmap, alpha=0.4, beta=1):
    img=copy.deepcopy(img_src)
    img = np.uint8(255 * img)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap= cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img * beta
    # scale 0 to 255  
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8) 
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    
    return superimposed_img

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
        heatmap = make_gradcam_heatmap(img, model, conv_layer_name, pred_index=None)
        img_grad_cam = superimpose_heatmap_on_image(img, heatmap, alpha=0.3)
        axs[i, 2].imshow(img_grad_cam)
        axs[i, 2].set_title("Grad-CAM", fontsize=12)
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])
        
        # Grad-CAM++
        heatmap_pp = grad_cam_plus(model, img, conv_layer_name)
        img_grad_cam_plus_plus = superimpose_heatmap_on_image(img, heatmap_pp, alpha=0.3)
        axs[i, 3].imshow(img_grad_cam_plus_plus)
        axs[i, 3].set_title("Grad-CAM++", fontsize=12)
        axs[i, 3].set_xticks([])
        axs[i, 3].set_yticks([])
    
        # Score-CAM
        heatmap_psc = ScoreCam(model, img, conv_layer_name)
        img_score_cam = superimpose_heatmap_on_image(img, heatmap_psc, alpha=0.3)
        axs[i, 4].imshow(img_score_cam)
        axs[i, 4].set_title("Score-CAM", fontsize=12)
        axs[i, 4].set_xticks([])
        axs[i, 4].set_yticks([])
    
    # Ajustar layout com espaçamento mínimo
    fig.subplots_adjust(wspace=0.1, hspace=0.2)  # Ajuste fino do espaçamento
    plt.tight_layout(pad=1)  # Ajustar o pad para margens controladas
    return fig

if __name__ == "__main__":
    #help(load_model)
    help(load_img_gen)
    help(predict_run) 
    help(grad_cam_plus)
    help(ScoreCam)    
    help(superimpose_heatmap_on_image)
    help(display_cam_grid)    