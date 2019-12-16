import os
import sys
import json

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.callbacks import ModelCheckpoint

from src.models.callbacks.EpochEvaluator import EpochEvaluator


class KerasModel():


    def __init__(self, model_name, config_path = 'config.json'):
        self.model_name = model_name
        self.model_type = model_name.split('_')[0]
        self.model = Sequential()
        
        if not self.read_config(config_path):
            print('Error: could not load config.json file')
            sys.exit(1)
        
        np.random.seed(self.config['np_rand_seed'])
        tf.random.set_seed(self.config['tf_rand_seed'])
        
        self.path = os.path.join(self.config['logs'], self.model_name)
        self.model_path = os.path.join(self.path, 'model.json')
        self.pred_path = os.path.join(self.path, 'predictions.csv')
        self.weights_path = os.path.join(self.path, 'weights.hdf5')
        self.epoch_eval_path = os.path.join(self.path, 'epoch_eval.json')
        
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        return
    
    
    def read_config(self, config_path):
        with open(config_path) as fr:
            self.config = json.load(fr)
            self.max_words = self.config['general']['max_words']
            self.config = self.config['KerasModel']
            self.config['trainable'] = self.config['trainable'] == 'True'
            return True
        
        return False
    
    
    def init_model(self, embeddings, params):
        self.params = params
        N, M = embeddings.shape
        self.model.add(Embedding(N, M, input_length = self.max_words, weights = [embeddings],
                                        name = 'embeddings', trainable = self.config['trainable']))
        
        if self.model_type == 'LSTM':
            self.model.add(LSTM(units = params['units'], dropout = params['dropout'],
                                recurrent_dropout = params['recurrent_dropout'], activation = params['activation']))
            self.model.add(Dense(100, activation = params['activation']))
            self.model.add(Dense(25, activation = params['activation']))
            self.model.add(Dense(50, activation = params['activation']))
        
        elif self.model_type == 'GRU':
            self.model.add(GRU(units = params['units'], dropout = params['dropout'],
                                recurrent_dropout = params['recurrent_dropout'], activation = params['activation']))
            self.model.add(Dense(150, activation = params['activation']))
            self.model.add(Dense(50, activation = params['activation']))
        
        elif self.model_type == 'LSTM+GRU':
            self.model.add(LSTM(units = params['units'], dropout = params['dropout'],
                                recurrent_dropout = params['recurrent_dropout'], return_sequences = True, activation = params['activation']))
            self.model.add(Dense(100, activation = params['activation']))
            self.model.add(Dense(50, activation = params['activation']))
            self.model.add(Dense(150, activation = params['activation']))
            self.model.add(GRU(units = params['units'], dropout = params['dropout'],
                                recurrent_dropout = params['recurrent_dropout'], return_sequences = True, activation = params['activation']))
            self.model.add(Dense(100, activation = params['activation']))
            self.model.add(Dense(50, activation = params['activation']))
            self.model.add(Dense(150, activation = params['activation']))
            self.model.add(LSTM(units = params['units'], dropout = params['dropout'],
                                recurrent_dropout = params['recurrent_dropout'], activation = params['activation']))
        
        else:
            print('Error: unknown model type')
            sys.exit(1)
        
        self.model.add(Dense(1, activation = params['final_activation']))
        
        
        if params['optimizer'] == 'RMSprop':
            optimizer = RMSprop(learning_rate = params['learning_rate'])
        
        elif params['optimizer'] == 'adam':
            optimizer = Adam(learning_rate = params['learning_rate'])
        
        elif params['optimizer'] == 'nadam':
            optimizer = Nadam(learning_rate = params['learning_rate'])
        
        
        self.model.compile(loss = params['loss'], optimizer = optimizer, metrics = params['metrics'])
        self.model.summary()
        with open(self.model_path, 'w') as fw:
            json.dump(self.model.to_json(), fw)
        
        return
    
    
    def predict(self, X, path = None, write = True):
        pred = self.model.predict_classes(X)
        
        if path == None:
            path = self.pred_path
        
        if write:
            with open(path, 'w', encoding = 'utf8') as fw:
                fw.write('Id,Prediction\n')
                
                for i in range(X.shape[0]):
                    fw.write('%d,%d\n' % (i + 1, 2 * pred[i] - 1))
            
        return pred
    
    
    def load(self, params, path = None):
        self.params = params
        
        if path == None:
            model_path = self.model_path
            weights_path = self.weights_path
        else:
            model_path = os.path.join(path, 'model.json')
            weights_path = os.path.join(path, 'weights.hdf5')
        
        with open(model_path, 'r') as fr:
            self.model = model_from_json(json.load(fr))
            self.model.load_weights(weights_path)
            self.model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = params['metrics'])
            self.model.summary()
            return True
        
        return False
    
    
    def train(self, tr, te, retrain = False):
        epoch_evaluator = EpochEvaluator(self.epoch_eval_path, retrain = retrain)
        checkpoint = ModelCheckpoint(filepath = self.weights_path, save_best_only = False, save_weights_only = True)
        
        self.model.fit(tr[0], tr[1], validation_data = te, epochs = self.params['epochs'],
                        batch_size = self.params['batch_size'], callbacks = [epoch_evaluator, checkpoint])
        return