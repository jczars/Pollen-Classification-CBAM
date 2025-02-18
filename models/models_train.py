"""
Module for training Keras models with additional functionalities, 
including callbacks and custom training loops.
"""

import datetime
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import EarlyStopping  # Removed unused imports
from keras.utils import custom_object_scope

def create_callbacks():
    """
    Creates a list of callbacks for training, including early stopping.

    Returns:
        list: A list of Keras callbacks.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True,
        mode='min'
    )
    return [early_stopping]

def run_train(train_data, val_data, model_fine, train_config):
    """
    Trains a Keras model, monitoring execution time and identifying the best epoch, with an option to display logs.

    Parameters:
        train_data: Training dataset.
        val_data: Validation dataset.
        model_fine: Model to be trained.
        train_config (dict): Dictionary containing training 
         configurations (batch_size, epochs, verbosity).

    Returns:
        history: Training history.
        start_time: Start time of the training.
        end_time: End time of the training.
        duration: Duration of the training.
        best_epoch: Best epoch based on validation loss.
    """
    batch_size = train_config['batch_size']
    epochs = train_config['epochs']
    verbosity = train_config.get('verbosity', 1)

    # Record start time
    start_time = datetime.datetime.now().replace(microsecond=0)
    if verbosity > 0:
        print(f'Batch size: {batch_size}\nTraining start time: {start_time}')

    # Configure callbacks and train on GPU
    with tf.device('/device:GPU:0'):
        print('\n', start_time)
        callbacks_list = create_callbacks()
        history = model_fine.fit(
            train_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1,
            validation_data=val_data
        )

    # End time and duration of training
    end_time = datetime.datetime.now().replace(microsecond=0)
    duration = end_time - start_time
    if verbosity > 0:
        print(f'Training duration: {duration}')

    # Identify the best epoch based on the lowest validation loss
    val_loss = history.history.get('val_loss', [])
    best_epoch = val_loss.index(min(val_loss)) + 1 if val_loss else None
    if verbosity > 0 and best_epoch:
        print(f'Best epoch: {best_epoch} with validation loss: {min(val_loss):.4f}')
    num_eapoch=len(history.history['loss'])

    return {
        'history':history, 
        'start_time': start_time, 
        'end_time': end_time, 
        'duration':duration, 
        'best_epoch': best_epoch,
        'num_epochs': num_eapoch
    }

def load_model(path_model, verbose=0):
    """
    Loads a Keras model from the specified path.

    Parameters:
        path_model (str): Path to the saved model.
        verbose (int): Verbosity level for model summary.

    Returns:
        model: The loaded Keras model.
    """
    model_rec = tf.keras.models.load_model(path_model)
    
    if verbose > 0:
        model_rec.summary()

    return model_rec

def load_model_vit(model_path, verbose=0):
    """
    Loads a Vision Transformer model from the specified path, using a custom optimizer if necessary.

    Parameters:
        model_path (str): Path to the saved model.
        verbose (int): Verbosity level for model summary.

    Returns:
        model: The loaded Vision Transformer model.
    """
    # Load the model with the registered optimizer in the custom object scope
    with custom_object_scope({'Addons>RectifiedAdam': tfa.optimizers.RectifiedAdam}):
        model = tf.keras.models.load_model(model_path)
    if verbose > 0:
        model.summary() 
    return model

if __name__ == "__main__":
    help(create_callbacks)
    help(run_train)
    help(load_model_vit)