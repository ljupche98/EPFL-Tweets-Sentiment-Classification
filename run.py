import os
import json

import pandas as pd

from src.serializer import *
from src.models.KerasModel import KerasModel


dfs = []

def run_model(path):
    config_path = os.path.join(path, 'config.json')
    with open(config_path) as fr:
        total_config = json.load(fr)
        config = total_config['general']
        config['full'] = config['full'] == 'True'
    
    raw_data_path = os.path.join(path, '../../../data/raw')
    
    generate_embeddings = True
    for file in os.listdir(raw_data_path):
        if file == 'X_%s_%d_avg.npy' % (config['emb_method'], config['emb_dim']):
            generate_embeddings = False
    
    if generate_embeddings:
        print('The required word embeddings not found')
        print('Starting to generate embeddings...')
        DataSerializer(full = config['full']) \
                .save_words(model = config['emb_method'], dim = config['emb_dim'], size = config['max_words'])
    else:
        print('The required word embeddings found')
    
    print('Starting to load them...')
    embeddings, X, y, X_test = DataDeserializer() \
                                .load_words(model = config['emb_method'], dim = config['emb_dim'], size = config['max_words'])
    
    model = KerasModel(config['model'], config_path = config_path)
    
    params = total_config['NN_hyperparams']
    params['metrics'] = [params['metrics']]
    model.load(params, path = path)
    
    pred_path = os.path.join(path, 'predictions.csv')
    model.predict(X_test, path = pred_path)
    
    global dfs
    df = pd.read_csv(os.path.join(path, 'predictions.csv'), index_col = ['Id'])
    dfs.append(df)
    
    return


path = os.getcwd()
path = os.path.join(path, 'logs/reproduce')

for model in os.listdir(path):
    if os.path.isdir(os.path.join(path, model)):
        run_model(os.path.join(path, model))

final_df = sum(df for df in dfs)
final_df['Prediction'].apply(lambda x: 1 if x > 0 else -1).reset_index().set_index('Id').to_csv(os.path.join(path, 'final_predictions.csv'))