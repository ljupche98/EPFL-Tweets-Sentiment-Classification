import numpy as np
from .utils.preprocessing import TweetPreprocessor


DATA_DIR = 'data/'


class TweetsModel:

    def load_tweets(self, fname):
        with open(fname, encoding = 'utf8') as file:
            return [ TweetPreprocessor.tokenize(line.rstrip()) for line in file.readlines() if len(line) ]

    def load_tweets_test(self, fname):
        with open(fname, encoding = 'utf8') as file:
            return [ TweetPreprocessor.tokenize(line.rstrip().split(',', 1)[1]) for line in file.readlines() if len(line) ]

    def __init__(self, full=False):
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

    def get_tweets(self):
        return self.tweets

    def get_labels(self):
        return self.labels

    def get_tweets_test(self):
        return self.tweets_test
