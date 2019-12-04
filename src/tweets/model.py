import numpy as np
from .utils.preprocessing import TweetPreprocessor


DATA_DIR = 'data/'


class TweetsModel:
    ''' This class is the tweets model. It holds all the tweets necessary
    to run the experiments.
    '''

    def load_tweets(self, fname):
        ''' Utility method for loading tweets from file.

        Parameters
        ----------
        fname: string
            The file name
        '''
        with open(fname, encoding = 'utf8') as file:
            return [ TweetPreprocessor.tokenize(line.rstrip()) for line in file.readlines() if len(line) ]

    def load_tweets_test(self, fname):
        ''' Utility method for loading the test tweets from file.
        The only difference between this and the previous method is
        that the test file contains the index at the begining of
        each line, which we have to ommit.

        Parameters
        ----------
        fname: string
            The file name
        '''
        with open(fname, encoding = 'utf8') as file:
            return [ TweetPreprocessor.tokenize(line.rstrip().split(',', 1)[1]) for line in file.readlines() if len(line) ]

    def build_vocab(self, tweets):
        ''' Utility method for building the vocaulary.

        Parameters
        ----------
        tweets: ndarray
            The tweets
        '''
        vocab = set()
        for tweet in tweets:
            vocab.update(tweet.split(' '))
        return vocab

    def __init__(self, full=False):
        ''' Constructs a TweetsModel object and commences the loading process.

        Parameters
        ----------
        full: boolean
            Whether to read the whole Twitter dataset
        '''
        ext = '_full.txt' if full else '.txt'
        print('Start loading train tweets...')
        tweets_pos = self.load_tweets(DATA_DIR + 'train_pos' + ext)
        print(' Loaded positive tweets...')
        tweets_neg = self.load_tweets(DATA_DIR + 'train_neg' + ext)
        print(' Loaded negative tweets...')
        self.tweets = np.array(tweets_pos + tweets_neg)
        print('Loading done!')
        self.labels = np.concatenate((np.ones(len(tweets_pos)), np.zeros(len(tweets_neg))))

        print('Start loading test tweets...')
        self.tweets_test = np.array(self.load_tweets_test(DATA_DIR + 'test_data.txt'))
        print('Loading done!')

        print('Start building vocab...')
        self.vocab = self.build_vocab(self.tweets)
        print('Vocab built!')

    def get_vocab(self):
        ''' Vocabulary getter '''
        return self.vocab

    def get_tweets(self):
        ''' Tweets getter '''
        return self.tweets

    def get_labels(self):
        ''' Labels getter '''
        return self.labels

    def get_tweets_test(self):
        ''' Test tweets getter '''
        return self.tweets_test
