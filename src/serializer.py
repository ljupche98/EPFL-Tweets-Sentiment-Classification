import numpy as np
import scipy.sparse as sp

from src.tweets.model import TweetsModel
from src.representations.model import *
from src.representations.controller import *
from src.representations.generator import *


RAW_DIR = 'data/raw/'


class DataSerializer:
    
    def __init__(self, full=False):
        self.tweets_model = TweetsModel(full)

    def save_generators(self, model='tf_idf', mode='word'):
        representations_generator = RepresentationsGenerator(self.tweets_model)
        if model == 'bow':
            X, test = representations_generator.bow()
            sp.save_npz(RAW_DIR + f'X_{model}', X)
            sp.save_npz(RAW_DIR + f'test_{model}', test)
        elif model == 'tf_idf':
            X, test = representations_generator.tf_idf(mode=mode)
            sp.save_npz(RAW_DIR + f'X_{model}_{mode}', X)
            sp.save_npz(RAW_DIR + f'test_{model}_{mode}', test)
        else:
            raise Exception('model unknown')
        y = self.tweets_model.get_labels()
        np.save(RAW_DIR + f'y_{model}', y)

    def save_words(self, model='glove', dim=25, size=50):
        representations_model = WordRepresentationsModel(model, dim, self.tweets_model.get_vocab())
        representations_controller = WordRepresentationsController(self.tweets_model, representations_model)
        representations = representations_controller.get_representations()
        X, test = representations_controller.get_representations_sequences(size)
        np.save(RAW_DIR + f'representations_{model}_{dim}', representations)
        np.save(RAW_DIR + f'X_{model}_{dim}_{size}_seq', X)
        np.save(RAW_DIR + f'test_{model}_{dim}_{size}_seq', test)
        X, test = representations_controller.get_representations_average()
        np.save(RAW_DIR + f'X_{model}_{dim}_avg', X)
        np.save(RAW_DIR + f'test_{model}_{dim}_avg', test)
        y = representations_controller.get_labels()
        np.save(RAW_DIR + f'y_{model}', y)

    def save_sentences(self, model='sent2vec', dim=25):
        representations_model = SentenceRepresentationsModel(model, dim)
        representations_controller = SentenceRepresentationsController(self.tweets_model, representations_model)
        X = representations_controller.get_representations()
        y = representations_controller.get_labels()
        np.save(RAW_DIR + f'X_{model}_{dim}', X)
        np.save(RAW_DIR + f'y_{model}', y)


class DataDeserializer:
    
    def load_generators(self, model='tf_idf', mode='word'):
        if model == 'bow':
            X = sp.load_npz(RAW_DIR + f'X_{model}.npz')
            test = sp.load_npz(RAW_DIR + f'test_{model}.npz')
        elif model == 'tf_idf':
            X = sp.load_npz(RAW_DIR + f'X_{model}_{mode}.npz')
            test = sp.load_npz(RAW_DIR + f'test_{model}_{mode}.npz')
        else:
            raise Exception('model unknown')
        y = np.load(RAW_DIR + f'y_{model}.npy', allow_pickle=True)
        return X, y, test

    def load_words(self, model='glove', mode='seq', dim=25, size=50):
        representations = np.load(RAW_DIR + f'representations_{model}_{dim}.npy', allow_pickle=True)
        y = np.load(RAW_DIR + f'y_{model}.npy', allow_pickle=True)
        if mode == 'seq':
            X = np.load(RAW_DIR + f'X_{model}_{dim}_{size}_{mode}.npy', allow_pickle=True)
            test = np.load(RAW_DIR + f'test_{model}_{dim}_{size}_{mode}.npy', allow_pickle=True)
            return representations, X, y, test
        elif mode == 'avg':
            X = np.load(RAW_DIR + f'X_{model}_{dim}_{mode}.npy', allow_pickle=True)
            test = np.load(RAW_DIR + f'test_{model}_{dim}_{mode}.npy', allow_pickle=True)
            return X, y, test
    
    def load_sentences(self, model='sent2vec', dim=25):
        X = np.load(RAW_DIR + f'X_{model}_{dim}.npy', allow_pickle=True)
        y = np.load(RAW_DIR + f'y_{model}.npy', allow_pickle=True)
        return X, y, None
