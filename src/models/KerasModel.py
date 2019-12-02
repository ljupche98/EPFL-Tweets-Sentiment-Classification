import os
import sys
import json

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from src.models.callbacks.EpochEvaluator import EpochEvaluator

from tensorflow.keras.models import model_from_json


class KerasModel():


    def __init__(self, model_type):
        self.model_type = model_type
        self.model = Sequential()
        
        if not self.read_config():
            print('Error: could not load config.json file')
            sys.exit(1)
        
        np.random.seed(self.config['np_rand_seed'])
        tf.random.set_seed(self.config['tf_rand_seed'])
        
        self.path = os.path.join(self.config['logs'], self.model_type)
        self.model_path = os.path.join(self.path, 'model.json')
        self.pred_path = os.path.join(self.path, 'predictions.csv')
        self.weights_path = os.path.join(self.path, 'weights.hdf5')
        self.epoch_eval_path = os.path.join(self.path, 'epoch_eval.json')
        # self.tensorboard_path = os.path.join(self.path, 'tensorboard_logs')
        
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        return
    
    
    def read_config(self):
        with open('config.json') as fr:
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
            self.model.add(LSTM(units = params['units'], dropout = params['dropout'], recurrent_dropout = params['recurrent_dropout']))
            self.model.add(Dense(1, activation = params['activation']))
            self.model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = params['metrics'])

        else:
            print('Error: unknown model type')
            sys.exit(1)
        
        self.model.summary()
        with open(self.model_path, 'w') as fw:
            json.dump(self.model.to_json(), fw)
        return
    
    
    def predict(self, X, write = True):
        pred = self.model.predict_classes(X)
        
        if write:
            with open(self.pred_path, 'w', encoding = 'utf8') as fw:
                fw.write('Id,Prediction\n')
                
                for i in range(len(X.shape[0])):
                    fw.write('%d,%d\n' % (i + 1, 2 * pred[i] - 1))
            
        return pred
    
    
    def load(self, params):
        self.params = params
        
        with open(self.model_path, 'r') as fr:
            self.model = model_from_json(json.load(fr))
            self.model.load_weights(self.weights_path)
            self.model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = params['metrics'])
            return True
        
        return False
    
    
    def train(self, tr, te, retrain = False):
        epoch_evaluator = EpochEvaluator(self.epoch_eval_path, retrain = retrain)
        # tensorboard = TensorBoard(log_dir = self.tensorboard_path)
        checkpoint = ModelCheckpoint(filepath = self.weights_path, save_best_only = True, save_weights_only = True)
        
        self.model.fit(tr[0], tr[1], validation_data = te, epochs = self.params['epochs'], batch_size = self.params['batch_size'],
                        callbacks = [epoch_evaluator, checkpoint]) #[epoch_evaluator, tensorboard, checkpoint])
        return


if __name__ == "__main__":
    LSTM_params = {
        'units': 100,
        'dropout': 0,
        'recurrent_dropout': 0,
        'activation': 'sigmoid',
        'loss': 'binary_crossentropy',
        'optimizer': 'RMSprop',
        'metrics': ['accuracy'],
        'epochs': 100,
        'batch_size': 128
    }


    test_model = KerasModel('LSTM')
    test_model.init_model(np.array([[0, 1], [2, 3], [4, 5]]), LSTM_params)
    test_model.train((np.array([[0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1])), (np.array([[0, 1]]), np.array([[0]])))


    test_model = KerasModel('LSTM')
    print(test_model.load(LSTM_params))