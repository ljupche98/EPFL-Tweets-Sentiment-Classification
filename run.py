import os
import json

import pandas as pd

from src.serializer import *
from src.models.KerasModel import KerasModel


# List of pd.DataFrame predictions from NN models
dfs = []

def run_model(path):
    ''' This function runs the model and makes the predictions for a given path to the folder.
    '''
    
    # Read the configuration file of the pre-trained model.
    config_path = os.path.join(path, 'config.json')
    with open(config_path) as fr:
        total_config = json.load(fr)
        config = total_config['general']
        config['full'] = config['full'] == 'True'
    
    # Calculate the path to the embeddings.
    raw_data_path = os.path.join(path, '../../../data/raw')
    
    # Check whether the required embeddings already exist.
    generate_embeddings = True
    for file in os.listdir(raw_data_path):
        if file == 'X_%s_%d_avg.npy' % (config['emb_method'], config['emb_dim']):
            generate_embeddings = False
    
    # If we have not got the required embeddings, generate them.
    if generate_embeddings:
        print('The required word embeddings not found')
        print('Starting to generate embeddings...')
        DataSerializer(full = config['full']) \
                .save_words(model = config['emb_method'], dim = config['emb_dim'], size = config['max_words'])
    else:
        print('The required word embeddings found')
    
    # Load the embeddings.
    print('Starting to load them...')
    embeddings, X, y, X_test = DataDeserializer() \
                                .load_words(model = config['emb_method'], dim = config['emb_dim'], size = config['max_words'])
    
    # Initialize the model.
    model = KerasModel(config['model'], config_path = config_path)
    
    # Load the pre-trained model.
    params = total_config['NN_hyperparams']
    params['metrics'] = [params['metrics']]
    model.load(params, path = path)
    
    # Predict the test data.
    pred_path = os.path.join(path, 'predictions.csv')
    model.predict(X_test, path = pred_path)
    
    # Append the predictions in the global variable
    global dfs
    df = pd.read_csv(os.path.join(path, 'predictions.csv'), index_col = ['Id'])
    dfs.append(df)
    
    return


# Initialize the path (get the current working directory).
path = os.getcwd()
path = os.path.join(path, 'logs/reproduce')

# For every folder in logs/reproduce, use the model to predict the test set.
for model in os.listdir(path):
    if os.path.isdir(os.path.join(path, model)):
        run_model(os.path.join(path, model))

# Calculate the final predictions as sum of the predictions (majority voting) and write them to a file.
final_df = sum(df for df in dfs)
final_df['Prediction'].apply(lambda x: 1 if x > 0 else -1).reset_index().set_index('Id').to_csv(os.path.join(path, 'final_predictions.csv'))