import numpy as np


DATA_DIR = 'data/representations/'


class WordRepresentationsModel:
    def load_representations(self, fname):
        mapping = {}
        representations = []

        with open(fname, encoding = 'utf8') as file:
             for i, line in enumerate(file.readlines()):
                tokens = line.rstrip().split(' ')
                mapping[tokens[0]] = i
                representations.append(eval(','.join(tokens[1:])))
        
        # For padding sequences
        representations.append([0] * self.dim)

        return mapping, np.array(representations)

    def __init__(self, src, dim):
        self.dim = dim
        self.mapping, self.representations = self.load_representations(DATA_DIR + src + f'.{str(dim)}d.txt')

    def get_dim(self):
        return self.dim

    def get_mapping(self):
        return self.mapping

    def get_representations(self):
        return self.representations


class SentenceRepresentationsModel:
    def load_representations(self, fname):
        representations = []

        with open(fname, encoding = 'utf8') as file:
             for i, line in enumerate(file.readlines()):
                tokens = line.rstrip().split(' ')
                representations.append(eval(','.join(tokens)))

        return np.array(representations)


    def __init__(self, src, dim):
        self.dim = dim
        self.representations = self.load_representations(DATA_DIR + src + f'.{str(dim)}d.txt')

    def get_dim(self):
        return self.dim

    def get_representations(self):
        return self.representations
