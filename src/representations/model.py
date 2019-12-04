import numpy as np


DATA_DIR = 'data/representations/'


class WordRepresentationsModel:
    ''' This class is the word representations model. It holds all the
    words in the tweets necessary to run the experiments.
    '''

    def load_representations(self, fname, vocab):
        ''' Utility function that loads the representations from file.
        It uses the vocabulary to filter unnecessary representations,
        grately reducing the representations matrix. Also, as side-effect
        generates a mapping from word to index, and returns it.

        Parameters
        ----------
        fname: string
            The filename
        vocab: set
            The vocabulary
        Returns
        -------
        tuple
            Mapping, representations
        '''
        mapping = {}
        representations = []
        i = 0
        with open(fname, encoding = 'utf8') as file:
             for line in file.readlines():
                tokens = line.rstrip().split(' ')
                if tokens[0] in vocab or tokens[0] == 'unk': # we need unk for unknown words
                    mapping[tokens[0]] = i
                    representations.append(eval(','.join(tokens[1:])))
                    i += 1
        # For padding sequences
        representations.append([0] * self.dim)
        return mapping, np.array(representations)

    def __init__(self, src, dim, vocab):
        ''' Constructs a WordRepresentationsModel object and loads the
        representations.

        Parameters
        ----------
        src: string
            The source model (glove or fasttext)
        dim: integer
            The representation dimension (25, 50, 100, or 200)
        vocab: set
            The vocabulary
        '''
        self.dim = dim
        self.mapping, self.representations = self.load_representations(DATA_DIR + src + f'.{str(dim)}d.txt', vocab)

    def get_dim(self):
        ''' Dimension getter '''
        return self.dim

    def get_mapping(self):
        ''' Mapping getter '''
        return self.mapping

    def get_representations(self):
        ''' Representations getter '''
        return self.representations


class SentenceRepresentationsModel:
    ''' This class is the sentence representations model. It holds all the
    tweets necessary to run the experiments.
    '''

    def load_representations(self, fname):
        ''' Utility function that loads the representations from file.
        It uses the vocabulary to filter unnecessary representations,
        grately reducing the representations matrix.

        Parameters
        ----------
        fname: string
            The filename
        Returns
        -------
        ndarray
            Representations
        '''
        representations = []
        with open(fname, encoding = 'utf8') as file:
             for i, line in enumerate(file.readlines()):
                tokens = line.rstrip().split(' ')
                representations.append(eval(','.join(tokens)))
        return np.array(representations)


    def __init__(self, src, dim):
        ''' Constructs a SentenceRepresentationsModel object and loads the
        representations.

        Parameters
        ----------
        src: string
            The source model (sent2vec)
        dim: integer
            The representation dimension (25, 50, 100, or 200)
        '''
        self.dim = dim
        self.representations = self.load_representations(DATA_DIR + src + f'.{str(dim)}d.txt')

    def get_dim(self):
        ''' Dimension getter '''
        return self.dim

    def get_representations(self):
        ''' Representations getter '''
        return self.representations
