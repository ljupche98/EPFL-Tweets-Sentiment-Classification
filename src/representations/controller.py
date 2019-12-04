import numpy as np
import functools
from tensorflow.keras.preprocessing.sequence import pad_sequences

def cache(func):
    ''' Keep a cache of previous function calls '''
    @functools.wraps(func)
    def wrapper_cache(*args, **kwargs):
        cache_key = func.__name__
        if cache_key not in wrapper_cache.cache:
            wrapper_cache.cache[cache_key] = func(*args, **kwargs)
        return wrapper_cache.cache[cache_key]
    wrapper_cache.cache = dict()
    return wrapper_cache


class RepresentationsController:
    ''' This class is the base representations controller. '''

    def __init__(self, sentences_model, representations_model):
        self.sentences_model = sentences_model
        self.representations_model = representations_model

    def get_representations(self):
        ''' Representations getter '''
        return self.representations_model.get_representations()

    def get_labels(self):
        ''' Labels getter '''
        return self.sentences_model.get_labels()


class WordRepresentationsController(RepresentationsController):
    ''' This class is the word representations controller. '''

    def map_words(self, sentence):
        ''' Maps the words to indices into the word representations
        matrix.

        Parameters
        ----------
        sentence: string
            The sentence to map
        Returns
        -------
        list
            The sequence
        '''
        mapping = self.representations_model.get_mapping()
        return list(map(lambda x: mapping[x] if x in mapping else mapping['unk'], sentence.split(' ')))

    def avg_words(self, sentence):
        ''' Averages the word representations.

        Parameters
        ----------
        sentence: string
            The sentence to average
        Returns
        -------
        list
            The averaged representations
        '''
        representations = self.representations_model.get_representations()
        sequence = self.map_words(sentence)
        return np.mean(representations[sequence], axis=0)

    def __init__(self, sentences_model, representations_model):
        ''' Constructs a WordRepresentationsController object. '''

        super().__init__(sentences_model, representations_model)
    
    def get_representations_sequences(self, size):
        ''' Representaion sequences getter.

        Parameters
        ----------
        size: integer
            The size of the padding, longer sequences are cut.
        Returns
        -------
        tuple
            Train sequences, test sequences
        '''
        def do(sentences):
            ''' Utility procedure that executes the sequnce generation '''
            representations = self.representations_model.get_representations()
            sequences = np.array([ self.map_words(sentence) for sentence in sentences ])
            return pad_sequences(sequences, maxlen=size, padding='post', value=len(representations) - 1)

        return do(self.sentences_model.get_tweets()), do(self.sentences_model.get_tweets_test())

    def get_representations_average(self):
        ''' Representaion average getter.

        Returns
        -------
        tuple
            Train sequences, test sequences
        '''
        def do(sentences):
            return np.array([ self.avg_words(sentence) for sentence in sentences ])

        return do(self.sentences_model.get_tweets()), do(self.sentences_model.get_tweets_test())


class SentenceRepresentationsController(RepresentationsController):
    ''' This class is the word representations controller. '''
    def __init__(self, sentences_model, representations_model):
        super().__init__(sentences_model, representations_model)
