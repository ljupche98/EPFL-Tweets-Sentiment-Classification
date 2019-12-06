import numpy as np
import subprocess
import fasttext
import sent2vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.tweets.utils.preprocessing import TweetPreprocessor


DATA_DIR = 'data/representations/'
TMP_DIR = 'tmp/'
MODELS_DIR = 'data/models/'


class RepresentationsGenerator:
    ''' This class is responsible for generating representations '''

    def __init__(self, sentences_model):
        ''' Constructs a RepresentationsGenerator object.

        Parameters
        ----------
        sentences_model: object
            The sentence model (ex.TweetsModel)
        '''
        self.sentences_model = sentences_model

    def bow(self):
        ''' Generates Bag-of-Words matrix.

        Note: Doesn't save the representations for performance reasons.

        Returns
        -------
        ndarray
        '''
        vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        vectorizer.fit(self.sentences_model.get_tweets())
        return vectorizer.transform(self.sentences_model.get_tweets()), vectorizer.transform(self.sentences_model.get_tweets_test())

    def tf_idf(self, mode='word'):
        ''' Generates TF-IDF matrix.
        
        Parameters
        ----------
        mode: string
            The mode (possible modes: 'word', 'ngram', and 'char')

        Note: Doesn't save the representations for performance reasons.

        Returns
        -------
        ndarray
        '''
        if mode == 'word':
            vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
        elif mode == 'ngram':
            vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        elif mode == 'char':
            vectorizer = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
        else:
            raise Exception('mode unknown')
        
        vectorizer.fit(self.sentences_model.get_tweets())
        return vectorizer.transform(self.sentences_model.get_tweets()), vectorizer.transform(self.sentences_model.get_tweets_test())

    def fasttext(self, **kwargs):
        ''' Generaters word represetations using fasttext and
        stores it into a file.

        Parameters
        ----------
        dim: integer
            The representation dimension
        load: boolean
            Whether to load an existing model
        '''
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
            model = fasttext.load_model(model_fname)
        else:
            raise Exception('load != True')

        words = model.words
        representations = [model.get_word_vector(x) for x in words]
        if 'unk' not in words: # if unk does not exist, embed it
            words.append('unk')
            representations.append(model.get_word_vector('unk'))
        data = np.concatenate((np.array(words).reshape(-1,1), np.array(representations)), axis=1)
        np.savetxt(DATA_DIR + f'fasttext.{str(dim)}d.txt', data, fmt='%s')
        
    def sent2vec(self, **kwargs):
        ''' Generaters sentence represetations using sent2vec and
        stores it into a file. Runs a subprocess of the compiled
        executable that reside on disk. You want to compile it yourself
        and put the executable in the root directory in order for this
        method to work.
        
        Follow the instructions in:
        https://github.com/epfml/sent2vec

        Parameters
        ----------
        dim: integer
            The representation dimension
        load: boolean
            Whether to load an existing model
        '''
        dim = kwargs['dim']

        tmp_fname = TMP_DIR + 'tweets.txt'
        model_fname = MODELS_DIR + f'sent2vec.{str(dim)}d.bin'

        if 'load' not in kwargs:
            np.savetxt(tmp_fname, self.sentences_model.get_tweets(), fmt='%s')
            subprocess.run(['./sent2vec/fasttext', 'sent2vec',
                            '-input', tmp_fname,
                            '-output', model_fname[:-4], # removes '.bin
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
