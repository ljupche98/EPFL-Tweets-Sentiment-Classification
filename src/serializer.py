import numpy as np
import scipy.sparse as sp

from src.tweets.model import TweetsModel
from src.representations.model import *
from src.representations.controller import *
from src.representations.generator import *


RAW_DIR = 'data/raw/'


class DataSerializer:
    ''' The purpose of this class is to generate train and test data and serialize
    it on hard disk for later loading. In that way, we save time when it's time for
    training a model.
    '''
    
    def __init__(self, full=False):
        ''' Constructs DataSerializer object.
        
        Parameters
        ----------
        full: boolean
            Whether to read the whole Twitter dataset
        '''
        self.tweets_model = TweetsModel(full)

    def save_generators(self, model='tf_idf', mode='word'):
        ''' Saves data directly obtained from the generator, that
        do not require training (BoW and TF-IDF).

        Parameters
        ----------
        model: string
            The model type (BoW or TF-IDF)
        mode: string
            The mode in the case of TF-IDF (word, ngram, or char)
        '''
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
        ''' Saves word representations as sequnces of indices to the
        embedding matrix as well as average word embeddings.

        Parameters
        ----------
        model: string
            The model type (glove or fasttext)
        dim: integer
            The representation dimension (25, 50, 100, or 200)
        size: integer
            In case of sequences, it is the length of the sequence
        '''
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
        ''' Saves sentence representations.

        Parameters
        ----------
        model: string
            The model type (currently only sent2vec is supported)
        dim: integer
            The represetation dimension
        '''
        representations_model = SentenceRepresentationsModel(model, dim)
        representations_controller = SentenceRepresentationsController(self.tweets_model, representations_model)
        X = representations_controller.get_representations()
        y = representations_controller.get_labels()
        np.save(RAW_DIR + f'X_{model}_{dim}', X)
        np.save(RAW_DIR + f'y_{model}', y)


class DataDeserializer:
    ''' The purpose of this class is to load the serialized train and test data
    from hard disk.
    '''
    
    def load_generators(self, model='tf_idf', mode='word'):
        ''' Loads data directly obtained from the generator, that
        do not require training (BoW and TF-IDF).

        Parameters
        ----------
        model: string
            The model type (BoW or TF-IDF)
        mode: string
            The mode in the case of TF-IDF (word, ngram, or char)

        Returns
        -------
        tuple
            Features, labels, and test set
        '''
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
        ''' lOADS word representations as sequnces of indices to the
        embedding matrix as well as average word embeddings.

        Parameters
        ----------
        model: string
            The model type (glove or fasttext)
        dim: integer
            The representation dimension (25, 50, 100, or 200)
        size: integer
            In case of sequences, it is the length of the sequence

        Returns
        -------
        tuple
            Features, labels, and test set
        '''
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
        ''' Loads sentence representations.

        Parameters
        ----------
        model: string
            The model type (currently only sent2vec is supported)
        dim: integer
            The represetation dimension

        Returns
        -------
        tuple
            Features, labels, and test set
        '''
        X = np.load(RAW_DIR + f'X_{model}_{dim}.npy', allow_pickle=True)
        y = np.load(RAW_DIR + f'y_{model}.npy', allow_pickle=True)
        return X, y, None
