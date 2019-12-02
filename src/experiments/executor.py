import time
from src.serializer import DataDeserializer
from sklearn.model_selection import train_test_split


class ExperimentExecutor:
    
    s = DataDeserializer()

    @staticmethod
    def get_data():
        yield 'BoW', ExperimentExecutor.s.load_generators('bow')
        yield 'Word level TF-IDF ', ExperimentExecutor.s.load_generators('tf_idf', 'word')
        yield 'N-Gram level TF-IDF', ExperimentExecutor.s.load_generators('tf_idf', 'ngram')
        yield 'Character level TF-IDF', ExperimentExecutor.s.load_generators('tf_idf', 'char')
        yield 'Pretrained GloVe embeddings, averaged', ExperimentExecutor.s.load_words('glove', 'avg')
        yield 'Trained FastText embeddings, averaged', ExperimentExecutor.s.load_words('fasttext', 'avg')
        yield 'Trained sent2vec embeddings', ExperimentExecutor.s.load_sentences()
    
    @staticmethod
    def prepare_data(X, y):
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @staticmethod
    def execute(experiment):
        for data in ExperimentExecutor.get_data():
            s = time.time()
            desc, (X, y, _) = data
            X_train, X_test, y_train, y_test = ExperimentExecutor.prepare_data(X, y)
            print(f'Execiting experiment "{experiment.__name__}" with "{desc}" data')
            experiment(X_train, X_test, y_train, y_test)
            print(f'Took: {round((time.time() - s) / 1000, 4)}s')
            print()
