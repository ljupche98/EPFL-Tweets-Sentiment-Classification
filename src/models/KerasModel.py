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
    ''' This class is the main class of the deep learning model. In the main use case,
    it takes as a parameter the type of the model ('LSTM', 'GRU', 'LSTM+GRU'). Additionally,
    an already pre-trained model can be loaded and the functions provided by this class can
    be used to train the model, save the weights and produce output.
    '''


    def __init__(self, model_name, config_path = 'config.json'):
        ''' Constructor of the object.
        
        Parameters
        ----------
        model_name: String
            Takes as input the type of the model. It can be in {'LSTM', 'GRU', 'LSTM+GRU'}.
        '''
        
        # Initialize the model name, type and Sequential type of Keras model.
        self.model_name = model_name
        self.model_type = model_name.split('_')[0]
        self.model = Sequential()
        
        # Try to load the configuration file of the model.
        if not self.read_config(config_path):
            print('Error: could not load config.json file')
            sys.exit(1)
        
        # Seet random seeds for reproducibility defined in the configuration file.
        np.random.seed(self.config['np_rand_seed'])
        tf.random.set_seed(self.config['tf_rand_seed'])
        
        # Initialize the default paths of the model with the given model name.
        self.path = os.path.join(self.config['logs'], self.model_name)
        self.model_path = os.path.join(self.path, 'model.json')
        self.pred_path = os.path.join(self.path, 'predictions.csv')
        self.weights_path = os.path.join(self.path, 'weights.hdf5')
        self.epoch_eval_path = os.path.join(self.path, 'epoch_eval.json')
        
        # Create a directory for the model if such does not exist.
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
        return
    
    
    def read_config(self, config_path):
        ''' Reads the configuration file of the model.
        
        Parameters
        ----------
        config_path: String
            Represents the path of the configuration of the model.
        
        Returns
        -------
        _: boolean
            Returns True if the configuration file was successfully read.
        '''
        
        # Read the configuration file and parse some boolean variables (because
        # JSON does not allow boolean variables in the format).
        with open(config_path) as fr:
            self.config = json.load(fr)
            self.max_words = self.config['general']['max_words']
            self.config = self.config['KerasModel']
            self.config['trainable'] = self.config['trainable'] == 'True'
            return True
        
        # Return False if the configuration file could not be read.
        return False
    
    
    def init_model(self, embeddings, params):
        ''' Initializes a new model.
        
        Parameters
        ----------
        embeddings: np.ndarray
            It is a matrix where every row represents an embedding of certain word.
        
        params: dictionary
            Additional parameters that the model may need.
        '''
        
        # Initialize the parameters of the architecture.
        self.params = params
        N, M = embeddings.shape
        
        # Add Embedding layer with the given matrix of word embeddings.
        self.model.add(Embedding(N, M, input_length = self.max_words, weights = [embeddings],
                                        name = 'embeddings', trainable = self.config['trainable']))
        
        # Create LSTM model.
        if self.model_type == 'LSTM':
            # Add LSTM layer with the given parameters.
            self.model.add(LSTM(units = params['units'], dropout = params['dropout'],
                                recurrent_dropout = params['recurrent_dropout'], activation = params['activation']))
            
            # Add multiple Dense layers. We can easily add multiple layers if we want so.
            self.model.add(Dense(100, activation = params['activation']))
            self.model.add(Dense(25, activation = params['activation']))
            self.model.add(Dense(50, activation = params['activation']))
        
        # Create GRU model.
        elif self.model_type == 'GRU':
            # Add GRU layer with the given parameters.
            self.model.add(GRU(units = params['units'], dropout = params['dropout'],
                                recurrent_dropout = params['recurrent_dropout'], activation = params['activation']))
            
            # Add multiple Dense layers. We can easily add multiple layers if we want so.
            self.model.add(Dense(150, activation = params['activation']))
            self.model.add(Dense(50, activation = params['activation']))
        
        # Create LSTM+GRU model.
        elif self.model_type == 'LSTM+GRU':
            # Add an initial layer (LSTM or GRU) with the given parameters.
            self.model.add(LSTM(units = params['units'], dropout = params['dropout'],
                                recurrent_dropout = params['recurrent_dropout'], return_sequences = True, activation = params['activation']))
            
            # Add multiple Dense layers. We can easily add multiple layers if we want so.
            self.model.add(Dense(100, activation = params['activation']))
            self.model.add(Dense(50, activation = params['activation']))
            self.model.add(Dense(150, activation = params['activation']))
            
            # Add another layer (LSTM or GRU) with the given parameters. We can easily add multiple layers if we want so.
            self.model.add(GRU(units = params['units'], dropout = params['dropout'],
                                recurrent_dropout = params['recurrent_dropout'], return_sequences = True, activation = params['activation']))
            
            # Add multiple Dense layers. We can easily add multiple layers if we want so.
            self.model.add(Dense(100, activation = params['activation']))
            self.model.add(Dense(50, activation = params['activation']))
            self.model.add(Dense(150, activation = params['activation']))
            
            # Add another layer (LSTM or GRU) with the given parameters. We can easily add multiple layers if we want so.
            self.model.add(LSTM(units = params['units'], dropout = params['dropout'],
                                recurrent_dropout = params['recurrent_dropout'], activation = params['activation']))
        
        # Error if we try to create a model we do not support.
        else:
            print('Error: unknown model type')
            sys.exit(1)
        
        # Add a final Dense layer used to get the prediction.
        self.model.add(Dense(1, activation = params['final_activation']))
        
        
        # Define an optimizer with the given parameters.
        # Create RMSprop optimizer.
        if params['optimizer'] == 'RMSprop':
            optimizer = RMSprop(learning_rate = params['learning_rate'])
        
        # Create Adam optimizer.
        elif params['optimizer'] == 'adam':
            optimizer = Adam(learning_rate = params['learning_rate'])
        
        # Create Nadam optimizer.
        elif params['optimizer'] == 'nadam':
            optimizer = Nadam(learning_rate = params['learning_rate'])
        
        
        # Compile the model and make sure everything is fine.
        self.model.compile(loss = params['loss'], optimizer = optimizer, metrics = params['metrics'])
        # Print the architecture of the model.
        self.model.summary()
        
        # Create a JSON file with the architecture of the model.
        with open(self.model_path, 'w') as fw:
            json.dump(self.model.to_json(), fw)
        
        return
    
    
    def predict(self, X, path = None, write = True):
        ''' Writes an output file with the preidctions of the model in the specified format.
        
        Parameters
        ----------
        X: np.ndarray
            Every row of the matrix represents one of the testing samples. Every column
            has a unique integer representation of the token in the raw textual format of the sample.
        
        path: String
            Specifies the location where the file should be written.
        
        write: boolean
            If True, write the predictions to file.
        
        Returns
        -------
        pred: np.ndarray
            The predictions of the model.
        '''
        
        # Get the predictions
        pred = self.model.predict_classes(X)
        
        # Use the default path if it is not specified.
        if path == None:
            path = self.pred_path
        
        # If write is True, write the predictions to the output file.
        if write:
            with open(path, 'w', encoding = 'utf8') as fw:
                fw.write('Id,Prediction\n')
                
                for i in range(X.shape[0]):
                    fw.write('%d,%d\n' % (i + 1, 2 * pred[i] - 1))
        
        return pred
    
    
    def load(self, params, path = None, load_weights = True):
        ''' Loads a pre-trained model.
        
        Parameters
        ----------
        params: dictionary
            The parameters of the pre-trained model.
        
        path: String
            The path where the model architecture and weights are.
        
        load_weights: boolean
            If True, the weights will be loaded. Else, only the architecture.
        
        Returns
        -------
            Returns True if the model was successfully read.
        '''
        
        # Update the parameters of the model
        self.params = params
        
        # If path is not specified, use the default paths.
        if path == None:
            model_path = self.model_path
            weights_path = self.weights_path
        
        # If path is specified, compute the new paths.
        else:
            model_path = os.path.join(path, 'model.json')
            weights_path = os.path.join(path, 'weights.hdf5')
        
        # Load the architecture (and / or weights) of the pre-trained model, print its summary and return True if successfully read.
        with open(model_path, 'r') as fr:
            self.model = model_from_json(json.load(fr))
            
            if load_weights:
                self.model.load_weights(weights_path)
            
            self.model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = params['metrics'])
            self.model.summary()

            return True
        
        return False
    
    
    def train(self, tr, te, retrain = False):
        ''' Trains a model for a given training and testing datasets.
        
        Parameters
        ----------
        tr: np.ndarray
            Training dataset where each row represents one tweet. Every column represents
            a unique ID of the token in the tweet.
        
        te: np.ndarray
            Testing dataset in the same format as the training dataset.
        
        retrain: boolean
            If True, it is considering as if the model was pre-trained.
        '''
        
        # Create callback classes.
        # Create EpochEvaluator class, used to log the performance metrics of the model.
        epoch_evaluator = EpochEvaluator(self.epoch_eval_path, retrain = retrain)
        # Create ModelCheckpoint class, used to save the weights of the model.
        checkpoint = ModelCheckpoint(filepath = self.weights_path, save_best_only = False, save_weights_only = True)
        
        # Train the model with the defined parameters.
        self.model.fit(tr[0], tr[1], validation_data = te, epochs = self.params['epochs'],
                        batch_size = self.params['batch_size'], callbacks = [epoch_evaluator, checkpoint])
        return