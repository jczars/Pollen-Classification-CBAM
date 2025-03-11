import sys
import gc, os
import pandas as pd

# Add the path for the modules
sys.path.append('/media/jczars/4C22F02A22F01B22/$WinREAgent/Pollen_classification_view/')
print(sys.path)

from models import get_data
from models import utils
from models import reports_build

def classificaImgs(conf, _path_model, _tempo, model_inst, _pseudo_csv,
                   CATEGORIES, tipo='png', verbose=1):
    """
    Classifies unlabeled images and generates pseudo-labels.

    Steps performed:
    1. Load model weights.
    2. Instantiate the model with the loaded weights.
    3. Load unlabeled images from the specified CSV.
    4. Generate pseudo-labels for the unlabeled images.
    5. Save the dataset of unlabeled images, including the timestamp.

    Parameters:
        conf: Configuration dictionary containing model parameters.
        _path_model: Path to the model weights.
        _tempo: Time variable for tracking iterations.
        model_inst: The model instance used for classification.
        _pseudo_csv: Path to the CSV containing pseudo-labels.
        CATEGORIES: List of image classes.
        tipo: Type of images ('png' by default).
        verbose: Verbosity level for debugging output.

    Returns:
        List of classified unlabeled data with pseudo-labels.
    """
    print('[classificaImgs].1 - Loading model weights')
    path = _path_model + conf['model'] + '_bestLoss_' + str(_tempo) + '.keras'
    print(path)

    print('[classificaImgs].2 - Instantiating the model with weights')
    model_inst.load_weights(path)

    print('[classificaImgs].3 - Loading unlabeled images from CSV')
    # Load unlabeled data
    if _tempo == 0:
        unalbels_generator = get_data.load_unlabels(conf)
    else:
        _cs_uns_ini = _pseudo_csv + '/unlabelSet_T' + str(_tempo) + '.csv'
        df_uns_ini = pd.read_csv(_cs_uns_ini)
        if len(df_uns_ini) > 0:
            print(df_uns_ini.head())
            print(f"\ntempo {_tempo}, read _cs_uns_ini{_cs_uns_ini}")
            unalbels_generator = get_data.load_data_test(df_uns_ini, input_size=(224, 224))
        else:
            return df_uns_ini

    print('[classificaImgs].4 - Generating pseudo-labels for unlabeled data')
    data_uns = reports_build.predict_unlabels_data_gen(conf, unalbels_generator, model_inst, CATEGORIES)
    print(f"data_uns, {len(data_uns)}")

    return data_uns


def selec(conf, data_uns_ini, _pseudo_csv, _tempo, train_data_csv, limiar):
    """
    Selects pseudo-labels from classified unlabeled data.

    Steps performed:
    1. Rename paths in the classified data.
    2. Filter the classified data by confidence level.
    3. Select pseudo-labels and exclude them from the original unlabeled dataset.
    4. Combine the previous training set with the selected pseudo-labels.
    5. Save the new training and unlabeled datasets.

    Parameters:
        conf (dict): Configuration settings for selection.
        data_uns_ini (DataFrame): Initial classified unlabeled data.
        _pseudo_csv (str): Path to save pseudo-labels.
        _tempo (int): Current iteration time.
        train_data_csv (DataFrame): Previous training dataset.
        limiar (float): Confidence threshold for selection.

    Returns:
        dict: Sizes of datasets and paths of new training set, or False if no labels selected.
    """
    
    # Step 1: Rename paths
    
    utils.renomear_path(conf, data_uns_ini)
    print("Initial Data Preview:", data_uns_ini.head())

    # Step 2: Filter by confidence level
    data_uns_fil = data_uns_ini[data_uns_ini['confidence'] > limiar]
    print(f'Filtered data size: {len(data_uns_fil)}')

    if data_uns_fil.empty:
        print('No pseudo-labels passed the confidence filter.')
        return False

    # Step 3: Exclude selected labels from the unlabeled dataset
    data_uns_ini = data_uns_ini[~data_uns_ini['file'].isin(data_uns_fil['file'])]
    print(f'Remaining unlabeled data size: {len(data_uns_ini)}')

    # Save the remaining unlabeled data
    tempo_px = _tempo + 1
    _csv_unlabels_t = os.path.join(_pseudo_csv, f'unlabelSet_T{tempo_px}.csv')
    print(f'Saving remaining unlabeled data to {_csv_unlabels_t}')
    data_uns_ini.to_csv(_csv_unlabels_t, index=False)

    # Step 4: Combine with previous training set
    
    if _tempo == 0:
        New_train_data = pd.concat([train_data_csv, data_uns_fil], ignore_index=True)
    else:
        previous_train_path = os.path.join(_pseudo_csv, f'trainSet_T{_tempo}.csv')
        train_data_csv = pd.read_csv(previous_train_path)
        New_train_data = pd.concat([train_data_csv, data_uns_fil], ignore_index=True)

    # Save the new training set
    _csv_New_TrainSet = os.path.join(_pseudo_csv, f'trainSet_T{tempo_px}.csv')
    print(f'Saving new training set to {_csv_New_TrainSet}')
    New_train_data.to_csv(_csv_New_TrainSet, index=False)

    # Return summary of selections and data sizes
    return {
        'ini': len(data_uns_ini) + len(data_uns_fil),
        'select': len(data_uns_fil),
        'rest': len(data_uns_ini),
        'train': len(train_data_csv),
        'new_train': len(New_train_data),
        '_csv_New_TrainSet': _csv_New_TrainSet
    }


if __name__ == "__main__": 
    help(classificaImgs)
    help(selec)
