import numpy as np
import functools
from tensorflow.keras.preprocessing.sequence import pad_sequences


def cache(func):
    """Keep a cache of previous function calls"""
    @functools.wraps(func)
    def wrapper_cache(*args, **kwargs):
        cache_key = func.__name__
        if cache_key not in wrapper_cache.cache:
            wrapper_cache.cache[cache_key] = func(*args, **kwargs)
        return wrapper_cache.cache[cache_key]
    wrapper_cache.cache = dict()
    return wrapper_cache


class RepresentationsController:
    def __init__(self, sentences_model, representations_model):
        self.sentences_model = sentences_model
        self.representations_model = representations_model

    def get_representations(self):
        return self.representations_model.get_representations()

    def get_labels(self):
        return self.sentences_model.get_labels()


class WordRepresentationsController(RepresentationsController):
    def map_words(self, sentence):
        mapping = self.representations_model.get_mapping()
        return list(map(lambda x: mapping[x] if x in mapping else mapping['<unknown>'], sentence.split(' ')))
    
    def avg_words(self, sentence):
        representations = self.representations_model.get_representations()
        sequence = self.map_words(sentence)
        return np.mean(representations[sequence], axis=0)
            
    def __init__(self, sentences_model, representations_model):
        super().__init__(sentences_model, representations_model)
    
    @cache
    def get_representations_sequences(self, size):
        sentences = self.sentences_model.get_tweets()
        mapping = self.representations_model.get_mapping()
        sequences = np.array([ self.map_words(sentence) for sentence in sentences ])
        return pad_sequences(sequences, maxlen=size, padding='post', value=len(mapping) - 1)
    
    @cache
    def get_representations_average(self):
        sentences = self.sentences_model.get_tweets()
        return np.array([ self.avg_words(sentence) for sentence in sentences ])


class SentenceRepresentationsController(RepresentationsController):
    def __init__(self, sentences_model, representations_model):
        super().__init__(sentences_model, representations_model)
