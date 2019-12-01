import numpy as np
from .utils.preprocessing import TweetPreprocessor


DATA_DIR = 'data/'


class TweetsModel:
    def load_tweets(self, fname):
        with open(fname) as file:
            return [ TweetPreprocessor.tokenize(line.rstrip()) for line in file.readlines() if len(line) ]

    def __init__(self, full=False):
        ext = '_full.txt' if full else '.txt'
        print('Start loading tweets...')
        tweets_pos = self.load_tweets(DATA_DIR + 'train_pos' + ext)
        print(' Loaded positive tweets...')
        tweets_neg = self.load_tweets(DATA_DIR + 'train_neg' + ext)
        print(' Loaded negative tweets...')
        self.tweets = np.array(tweets_pos + tweets_neg)
        print('Loading done!')
        self.labels = np.concatenate((np.ones(len(tweets_pos)), np.zeros(len(tweets_neg))))

    def get_tweets(self):
        return self.tweets

    def get_labels(self):
        return self.labels
