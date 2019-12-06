import time
from src.serializer import DataDeserializer
from sklearn.model_selection import train_test_split


class ExperimentExecutor:
    ''' This class is responsible for executing the experiments
    for the classical ML models.
    '''
    
    s = DataDeserializer()

    @staticmethod
    def get_data():
        ''' Data generator. '''
        yield 'BoW', ExperimentExecutor.s.load_generators('bow')
        yield 'Word level TF-IDF ', ExperimentExecutor.s.load_generators('tf_idf', 'word')
        yield 'N-Gram level TF-IDF', ExperimentExecutor.s.load_generators('tf_idf', 'ngram')
        yield 'Character level TF-IDF', ExperimentExecutor.s.load_generators('tf_idf', 'char')
        yield 'Pretrained GloVe embeddings, averaged', ExperimentExecutor.s.load_words('glove', 'avg', 100)
        yield 'Trained FastText embeddings, averaged', ExperimentExecutor.s.load_words('fasttext', 'avg', 100)
        yield 'Trained sent2vec embeddings', ExperimentExecutor.s.load_sentences('sent2vec', 100)
    
    @staticmethod
    def prepare_data(X, y):
        ''' Performs train-test split'''
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @staticmethod
    def execute(experiment, load=False):
        ''' This method executes an experiment.

        1. It requres function that accepts: X_train, X_test, y_train, y_test datasets
        in that particular order.
        2. Further, it assumes that the experiment does a propper grid search over the
        hyperparameter space and does CV to measure the accuracy.
        3. For every possible dataset, it executes the experiment, printing the results.
        4. Optionally, persists the model

        Parameters
        ----------
        experiment: function
            The experiment to execute
        load: boolean
            Whether to load model
        '''
        for data in ExperimentExecutor.get_data():
            s = time.time()
            desc, (X, y, _) = data
            X_train, X_test, y_train, y_test = ExperimentExecutor.prepare_data(X, y)
            print(f'Execiting experiment "{experiment.__name__}" with "{desc}" data')
            experiment(X_train, X_test, y_train, y_test, load)
            print(f'Took: {round((time.time() - s))}s')
            print()
