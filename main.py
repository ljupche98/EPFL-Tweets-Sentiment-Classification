import numpy as np
np.set_printoptions(suppress=True)

# from src.serializer import DataSerializer
# s = DataSerializer()
# s.save_generators('bow')
# s.save_generators('tf_idf', 'word')
# s.save_generators('tf_idf', 'ngram')
# s.save_generators('tf_idf', 'char')
# s.save_words('glove')
# s.save_words('fasttext')
# s.save_sentences()

# from src.tweets.model import TweetsModel
# from src.representations.generator import RepresentationsGenerator
# tweets_model = TweetsModel()
# representations_generator = RepresentationsGenerator(tweets_model)
# representations_generator.sent2vec(dim=25)

from src.experiments.executor import ExperimentExecutor
from src.models.classic import LogisticRegression
ExperimentExecutor.execute(LogisticRegression)
