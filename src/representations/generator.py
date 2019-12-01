import numpy as np
import subprocess
import fasttext
import sent2vec
from src.tweets.utils.preprocessing import TweetPreprocessor


DATA_DIR = 'data/representations/'
TMP_DIR = 'tmp/'
MODELS_DIR = 'data/models/'


class RepresentationsGenerator:
    def __init__(self, sentences_model=None):
        self.sentences_model = sentences_model

    def bow(self):
        pass

    def tf_idf(self):
        pass

    def fasttext(self, **kwargs):
        if not self.sentences_model:
            raise Exception('sentences_model == None')

        dim = kwargs['dim']

        tmp_fname = TMP_DIR + 'tweets.txt'
        model_fname = MODELS_DIR + f'fasttext.{str(dim)}d.bin'

        if 'load' not in kwargs:
            np.savetxt(tmp_fname, self.sentences_model.get_tweets(), fmt='%s')
            model = fasttext.train_unsupervised(tmp_fname, model='skipgram',
                                                minCount=1,
                                                dim=dim)
            model.save_model(model_fname)
        elif kwargs['load']:
            model = fasttext.load_model(model_name)
        else:
            raise Exception('load != True')
        
        representations = np.array([model.get_word_vector(x) for x in model.words])
        data = np.concatenate((np.array(model.words).reshape(-1,1), representations), axis=1)
        np.savetxt(DATA_DIR + f'fasttext.{str(dim)}d.txt', data, fmt='%s')
        
    def sent2vec(self, **kwargs):
        if not self.sentences_model:
            raise Exception('sentences_model == None')

        dim = kwargs['dim']

        tmp_fname = TMP_DIR + 'tweets.txt'
        model_fname = MODELS_DIR + f'sent2vec.{str(dim)}d.bin'

        if 'load' not in kwargs:
            np.savetxt(tmp_fname, self.sentences_model.get_tweets(), fmt='%s')
            subprocess.run(['./fasttext', 'sent2vec',
                            '-input', tmp_fname,
                            '-output', model_fname[:-4], # remove '.bin
                            '-minCount', '1',
                            '-dim', str(dim)])
            model = sent2vec.Sent2vecModel()
            model.load_model(model_fname)
        elif kwargs['load']:
            model = sent2vec.Sent2vecModel()
            model.load_model(model_fname)
        else:
            raise Exception('load != True')
        
        representations = model.embed_sentences(self.sentences_model.get_tweets())
        np.savetxt(DATA_DIR + f'sent2vec.{str(dim)}d.txt', representations, fmt='%s')
