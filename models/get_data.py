# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split

def aug_param(_aug):
    #Augmentation
    print('\n ######## Data Generator ################')
    #https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

    if _aug=='sem':
        aug=dict(rescale=1./255)
    if _aug== 'aug0':
        aug=dict(rescale=1./255,
                 brightness_range=[0.2,0.8],
                 horizontal_flip = True)
    if _aug== 'aug1':
        aug=dict(width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.3,
        fill_mode='nearest',
        horizontal_flip=True,
        rescale=1./255,
        )

    idg = ImageDataGenerator(**aug)
    return idg

def load_data_train_aug_param(training_data, val_data, aug, input_size):
  """
    -->loading train data
    :param: training_data: ptah at dataset
    :param: BATCH: batch size
    :param: INPUT_SIZE: input dimensions, height and width, default=(224,224)
    :param: SPLIT_VALID: portion to divide the training base into training and validation
    return: train and valid dataset
    """
  idg = aug_param(aug)

  train_data_generator = idg.flow_from_dataframe(training_data,
                                            x_col = "file",
                                            y_col = "labels",
                                            target_size=input_size,
                                            class_mode = "categorical",
                                            shuffle = True)
  valid_data_generator = idg.flow_from_dataframe(val_data,
                                                x_col = "file",
                                                y_col = "labels",
                                                target_size=input_size,
                                                class_mode = "categorical",
                                                shuffle = True)
  return train_data_generator, valid_data_generator

def reload_data_train(conf, _csv_training_data, SPLIT_VALID=0.2):
    """
    -->loading train data
    :param: training_data: ptah at dataset
    :param: BATCH: batch size
    :param: INPUT_SIZE: input dimensions, height and width, default=(224,224)
    :param: SPLIT_VALID: portion to divide the training base into training and validation
    return: train and valid dataset
    """
    print('training_data_path ',_csv_training_data)
    training_data=pd.read_csv(_csv_training_data)
    idg = aug_param(conf['aug'])

    idg = ImageDataGenerator(rescale=1. / 255, validation_split=SPLIT_VALID)
    img_size=conf['img_size']
    input_size=(img_size, img_size)

    train_generator = idg.flow_from_dataframe(
        dataframe=training_data,
        x_col = "file",
        y_col = "labels",
        target_size=input_size,
        class_mode = "categorical",
        shuffle = True,
        subset='training')

    val_generator = idg.flow_from_dataframe(
        dataframe=training_data,
        x_col = "file",
        y_col = "labels",
        target_size=input_size,
        class_mode = "categorical",
        shuffle = True,
        subset='validation')

    return train_generator, val_generator

def load_data_train_aug(PATH_BD, K, BATCH, INPUT_SIZE, SPLIT_VALID):
    """
    Load training data with augmentation.
    
    Args:
        PATH_BD (str): Path to the dataset.
        K (int): K-fold value.
        BATCH (int): Batch size.
        INPUT_SIZE (tuple): Input dimensions (height, width).
        SPLIT_VALID (float): Portion to split the training data into training and validation sets.
        
    Returns:
        tuple: Training and validation datasets.
    """
    train_dir = PATH_BD + '/Train/k' + str(K)
    print('Training directory: ', train_dir)
    
    idg = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.3,
        fill_mode='nearest',
        horizontal_flip=True,
        rescale=1./255,
        validation_split=SPLIT_VALID
    )

    train_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='validation'
    )

    return train_generator, val_generator

def load_data_train(PATH_BD, K, BATCH, INPUT_SIZE, SPLIT_VALID):
    """
    Load training data with augmentation.
    
    Args:
        PATH_BD (str): Path to the dataset.
        K (int): K-fold value.
        BATCH (int): Batch size.
        INPUT_SIZE (tuple): Input dimensions (height, width).
        SPLIT_VALID (float): Portion to split the training data into training and validation sets.
        
    Returns:
        tuple: Training and validation datasets.
    """
    train_dir = PATH_BD + '/Train/k' + str(K)
    print('Training directory: ', train_dir)
    
    idg = ImageDataGenerator(
        rescale=1./255,
        validation_split=SPLIT_VALID
    )

    train_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42,
        subset='validation'
    )

    return train_generator, val_generator

def load_data_ttv(PATH_BD, K, BATCH, INPUT_SIZE, SPLIT_VALID):
    """
    Load training and validation data.
    
    Args:
        PATH_BD (str): Path to the dataset.
        K (int): K-fold value.
        BATCH (int): Batch size.
        INPUT_SIZE (tuple): Input dimensions (height, width).
        SPLIT_VALID (float): Portion to split the training data into training and validation sets.
        
    Returns:
        tuple: Training and validation datasets.
    """
    train_dir = PATH_BD + '/Train/k' + str(K)
    val_dir = PATH_BD + '/Val/k' + str(K)
    print('Training directory: ', train_dir)
    
    idg = ImageDataGenerator(rescale=1. / 255)

    train_generator = idg.flow_from_directory(
        directory=train_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    val_generator = idg.flow_from_directory(
        directory=val_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    return train_generator, val_generator

def load_data_test(test_data, input_size):
  idg = ImageDataGenerator(rescale=1. / 255)
  test_data_generator = idg.flow_from_dataframe(test_data,
                                            x_col = "file",
                                            y_col = "labels",
                                            target_size=input_size,
                                            class_mode = "categorical",
                                            shuffle = False)
  return test_data_generator

def load_data_test_dir(PATH_BD, K, BATCH, INPUT_SIZE):
    """
    -->loading train data 
    :param: PATH_BD: file name 
    :param: K: k the kfolders values
    :param: BATCH: batch size
    :param: INPUT_SIZE: input dimensions, height and width, default=(224,224)
    :return: test dataset
    """
    test_dir = PATH_BD + '/Test/k' + str(K)
    print('test_dir ', test_dir)

    idg = ImageDataGenerator(rescale=1. / 255)
    test_generator = idg.flow_from_directory(
        directory=test_dir,
        target_size=INPUT_SIZE,
        color_mode="rgb",
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False,
        seed=42)
    return test_generator
                             
def load_unlabels(conf):
    
  idg = ImageDataGenerator(rescale=1. / 255)
  path=str(conf['unlabels'])
  print("conf['unlabels']: ", str(path))
  
  img_size=conf['img_size']
  input_size=(img_size, img_size)
  print('input_size ',(img_size, img_size))

  unalbels_generator = idg.flow_from_directory(
      directory=path,
      target_size=(input_size),
      color_mode="rgb",
      batch_size=conf['batch_size'],
      class_mode=None,
      shuffle=False,
      seed=42)
  return unalbels_generator

def splitData(data_csv, path_save, name_base):
  """
  -->Split dataSet into training and testing data
  :param: data_csv: dataSet in csv format
  :param: path_save: path to save training and testing data
  :param: name_base: name to save the data
  """

  prod_csv, test_csv = train_test_split(data_csv, test_size=0.2, shuffle=True)
  train_csv, val_csv = train_test_split(prod_csv, test_size=0.2, shuffle=True)
  #print('Train ',train_csv.shape)
  #print('Test ',test_csv.shape)

  #Salvar split data
  _path_train=path_save+'/'+name_base+'_trainSet.csv'
  _path_test=path_save+'/'+name_base+'_testSet.csv'
  _path_val=path_save+'/'+name_base+'_valSet.csv'
  train_csv.to_csv(_path_train, index = False, header=True)
  test_csv.to_csv(_path_test, index = False, header=True)
  val_csv.to_csv(_path_val, index = False, header=True)

  training_data = pd.read_csv(_path_train)
  print('\n Train split')
  print(training_data.groupby('labels').count())
  test_data = pd.read_csv(_path_test)
  print('\n Test split')
  print(test_data.groupby('labels').count())

  val_data = pd.read_csv(_path_val)
  print('\n Val split')
  print(val_data.groupby('labels').count())

  return _path_train,_path_val,_path_test

if __name__=="__main__":
   help(load_data_train)
   help(reload_data_train)
   help(load_data_test)
   help(load_unlabels)
   help(splitData)