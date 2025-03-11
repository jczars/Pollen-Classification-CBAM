from keras import Model
import keras, sys
from keras.layers import Dense, Input
from keras import layers
from keras.applications import ResNet50, MobileNet, DenseNet201, InceptionV3
from keras.applications import ResNet152V2, Xception, VGG16, VGG19

sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
from models import utils

def print_layer(conv_model, layers_params, verbose=0):
    """
    Prints and saves information about the layers of a convolutional model.

    Parameters
    ----------
    conv_model : keras.Model
        The convolutional model whose layer details will be printed and optionally saved.
    layers_params : dict
        Dictionary containing parameters for layer information storage, including:
            - 'save_dir' : str
                Directory path where the layer information will be saved.
            - 'id_test' : str
                Unique identifier for the test or model instance.
            - 'model' : str
                Model name to include in the saved file.

    Returns
    -------
    None
        This function prints layer details and, if a directory is specified, saves it to a CSV file.
    
    Notes
    -----
    If 'save_dir' is provided, this function creates a directory structure in 'save_dir/models/' 
    and saves the CSV file named '<id_test>_<model>_layers.csv'. The file includes the 
    trainable status and name of each layer in the model.
    """
    # Define save directory based on parameters
    save_dir = layers_params['save_dir']
    # Create a model name using test ID and model name from parameters
    nm_model = f"{layers_params['id_test']}_{layers_params['model']}"
    
    if save_dir:
        # Specify the full path to the model save directory
        #save_dir = f"{save_dir}/{nm_model}/"
        if verbose>0:
            print('save_dir ', save_dir)
        # Create necessary folders in the specified save directory
        utils.create_folders(save_dir, flag=0)
        
        # Define CSV path for saving layer information
        _csv_layers = save_dir + '/' + nm_model + '_layers.csv'
        
        # Add initial test ID and model name to the CSV
        utils.add_row_csv(_csv_layers, [['id_test', layers_params['id_test']]])
        utils.add_row_csv(_csv_layers, [['model', layers_params['model']]])
        utils.add_row_csv(_csv_layers, [['trainable', 'name']])

    # Initialize an array to store layer details
    layers_arr = []
    # Iterate through each layer in the model
    for i, layer in enumerate(conv_model.layers):
        # Print the layer index, trainable status, and name
        if verbose>0:
            print("{0} {1}:\t{2}".format(i, layer.trainable, layer.name))
        layers_arr.append([layer.trainable, layer.name])
    
    # Save layer details to CSV if save directory was specified
    if save_dir:
        utils.add_row_csv(_csv_layers, layers_arr)


def hyper_model(config_model, verbose=0):
    """
    Builds and configures a fine-tuned model based on a pre-trained base model.

    Parameters
    ----------
    config_model : dict
        Configuration dictionary with the following keys:
            - 'model' : str
                Name of the pre-trained model to be used (e.g., "VGG16").
            - 'id_test' : str
                Identifier for the test or specific model instance.
            - 'num_classes' : int
                Number of output classes for the classification task.
            - 'last_activation' : str
                Activation function for the final dense layer (e.g., "softmax").
            - 'freeze' : int
                Number of layers to freeze in the base model for transfer learning.
            - 'save_dir' : str
                Path to the directory where layer information will be saved.
            - 'optimizer' : str
                Name of the optimizer to use (e.g., "Adam", "SGD", "RMSprop", "Adagrad").
            - 'learning_rate' : float
                Learning rate for the optimizer.

    Returns
    -------
    keras.Model
        Compiled Keras model with fine-tuning and the specified configuration.

    Notes
    -----
    This function loads a pre-trained model (with 'imagenet' weights), freezes a certain number
    of layers as specified, adds a custom dense layer for classification, and optionally unfreezes
    some layers for further fine-tuning. The optimizer and learning rate are also set according 
    to the provided configuration.

    Documentation Style
    -------------------
    - Function documentation follows the **NumPy style** for readability and structured presentation.
    - Code is refactored according to **PEP8** coding standards for Python, focusing on readability,
      modularity, and clear comments.
    """
    
    # Initialize the specified pre-trained model
    model = eval(config_model['model'])
    id_test = config_model['id_test']
    
    # Load the base model with 'imagenet' weights and no top layer
    base_model = model(include_top=True, weights='imagenet')

    # Freeze all layers in the base model initially
    for layer in base_model.layers:
        layer.trainable = False
        if verbose>0:
            base_model.summary()

    # Connect a custom output layer to the base model
    conv_output = base_model.layers[-2].output  # Use the second last layer as output
    output = layers.Dense(config_model['num_classes'], name='predictions', 
                          activation=config_model['last_activation'])(conv_output)
    fine_model = Model(inputs=base_model.input, outputs=output)

    # Unfreeze layers based on the 'freeze' index for fine-tuning
    freeze = config_model['freeze']
    for layer in base_model.layers[freeze:]:
        layer.trainable = True

    # Prepare parameters for saving layer information
    layers_params = {'id_test': id_test, 'save_dir': config_model['save_dir'], 'model': config_model['model']}
    
    # Print and save layer details
    if verbose>0:
        print_layer(fine_model, layers_params, 1)

    # Set the optimizer based on configuration
    optimizer_name = config_model['optimizer']
    learning_rate = config_model['learning_rate']
    if optimizer_name == 'Adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSprop':
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'Adagrad':
        opt = keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'SGD':
        opt = keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    if verbose>0:
        print(opt)

    # Compile the model with categorical crossentropy loss and accuracy metric
    fine_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    if verbose>0:
        fine_model.summary()
    return fine_model

def hyper_model_up(config_model, verbose=0):
    """
    Builds and configures a fine-tuned model based on a pre-trained base model.

    Parameters
    ----------
    config_model : dict
        Configuration dictionary with the following keys:
            - 'model' : str
                Name of the pre-trained model to be used (e.g., "VGG16").
            - 'id_test' : str
                Identifier for the test or specific model instance.
            - 'num_classes' : int
                Number of output classes for the classification task.
            - 'last_activation' : str
                Activation function for the final dense layer (e.g., "softmax").
            - 'freeze' : int
                Number of layers to freeze in the base model for transfer learning.
            - 'save_dir' : str
                Path to the directory where layer information will be saved.
            - 'optimizer' : str
                Name of the optimizer to use (e.g., "Adam", "SGD", "RMSprop", "Adagrad").
            - 'learning_rate' : float
                Learning rate for the optimizer.

    Returns
    -------
    keras.Model
        Compiled Keras model with fine-tuning and the specified configuration.
    
    Notes
    -----
    This function loads a pre-trained model (with 'imagenet' weights), freezes a certain number
    of layers as specified, adds a custom dense layer for classification, and optionally unfreezes
    some layers for further fine-tuning. The optimizer and learning rate are also set according 
    to the provided configuration.
    """
    
    # Load the specified pre-trained model
    model_class = globals().get(config_model['model'], None)
    if model_class is None:
        raise ValueError(f"Model {config_model['model']} not found.")
    
    # Load the base model with 'imagenet' weights and without the top layer
    img_size = config_model['img_size']
    input_shape = (img_size, img_size, 3)
    base_model = model_class(include_top=False, weights='imagenet', input_shape=input_shape)

    # Freeze layers based on the freeze index
    for i, layer in enumerate(base_model.layers):
        layer.trainable = i >= config_model['freeze']  # Freeze layers up to the 'freeze' index

    # Add custom output layer for classification
    conv_output = base_model.output
    conv_output = layers.GlobalAveragePooling2D()(conv_output)
    output = layers.Dense(config_model['num_classes'], name='predictions', 
                          activation=config_model['last_activation'])(conv_output)
    fine_model = Model(inputs=base_model.input, outputs=output)

    # Save layer details
    layers_params = {'id_test': config_model['id_test'], 
                     'save_dir': config_model['save_dir'], 
                     'model': config_model['model']}
    
    if verbose > 0:
        # Print layer details
        print_layer(fine_model, layers_params, 0)

    # Configure the optimizer
    optimizer_dict = {
        'Adam': keras.optimizers.Adam,
        'RMSprop': keras.optimizers.RMSprop,
        'Adagrad': keras.optimizers.Adagrad,
        'SGD': keras.optimizers.SGD
    }
    optimizer_name = config_model['optimizer']
    optimizer_class = optimizer_dict.get(optimizer_name, None)
    if optimizer_class is None:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")
    
    opt = optimizer_class(learning_rate=config_model['learning_rate'])
    
    if verbose > 0:
        print(opt)

    # Compile the model
    fine_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    if verbose > 0:
        fine_model.summary()
    
    return fine_model



