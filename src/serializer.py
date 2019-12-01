from .tweets.model import TweetsModel
from .representations.model import *
from .representations.controller import *
from .representations.generator import *


RAW_DIR = 'data/raw/'


class DataSerializer:
    
    def __init__(self, full=False):
        self.tweets_model = TweetsModel(full)

    def save_generators(self, model='tf_idf'):
        representations_generator = RepresentationsGenerator(self.tweets_model)
        if model == 'bow':
            X = representations_generator.bow()
        elif model == 'tf_idf':
            X = representations_generator.tf_idf(mode='word')
        else:
            raise Exception('model unknown')
        y = self.tweets_model.get_labels()
        np.save(RAW_DIR + f'X_{model}', X)
        np.save(RAW_DIR + f'y_{model}', y)

    def save_words(self, model='glove', dim='50', size=50):
        representations_model = WordRepresentationsModel(model, dim)
        representations_controller = WordRepresentationsController(self.tweets_model, representations_model)
        representations = representations_controller.get_representations()
        X = representations_controller.get_representations_sequences(size)
        y = representations_controller.get_labels()
        np.save(RAW_DIR + f'representations_{model}_{dim}_{size}', representations)
        np.save(RAW_DIR + f'X_{model}_{dim}_{size}', X)
        np.save(RAW_DIR + f'y_{model}_{dim}_{size}', y)

    def save_sentences(self, model='sent2vec', dim='50'):
        representations_model = SentenceRepresentationsModel(model, dim)
        representations_controller = SentenceRepresentationsController(self.tweets_model, representations_model)
        X = representations_controller.get_representations()
        y = representations_controller.get_labels()
        np.save(RAW_DIR + f'X_{model}_{dim}', X)
        np.save(RAW_DIR + f'y_{model}_{dim}', y)


class DataDeserializer:
    
    def load_generators(self, model='tf_idf'):
        X = np.load(RAW_DIR + f'X_{model}.npy', allow_pickle=True)
        y = np.load(RAW_DIR + f'y_{model}.npy', allow_pickle=True)
        return X, y

    def load_words(self, model='glove', dim='50', size=50):
        representations= np.load(RAW_DIR + f'representations_{model}_{dim}_{size}.npy', allow_pickle=True)
        X = np.load(RAW_DIR + f'X_{model}_{dim}_{size}.npy', allow_pickle=True)
        y = np.load(RAW_DIR + f'y_{model}_{dim}_{size}.npy', allow_pickle=True)
        return representations, X, y

    def load_sentences(self, model='sent2vec', dim='50', allow_pickle=True):
        X = np.load(RAW_DIR + f'X_{model}_{dim}.npy', allow_pickle=True)
        y = np.load(RAW_DIR + f'y_{model}_{dim}.npy', allow_pickle=True)
        return X, y
