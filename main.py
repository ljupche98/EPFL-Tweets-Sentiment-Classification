''' The main script for executing classical ML experiments.

Uncomment the lines that you want to execute.
1. It is necessary to have representations on disk
2. It is necessary to have train and test data on disk
3. If 1. and 2. are satisfied, just execute the experiments
'''
import numpy as np
np.set_printoptions(suppress=True)

# ------------------------------------------------------------------
# REPRESENTATIONS GENERATION
# ------------------------------------------------------------------
# from src.tweets.model import TweetsModel
# from src.representations.generator import RepresentationsGenerator
# tweets_model = TweetsModel()
# representations_generator = RepresentationsGenerator(tweets_model)
# representations_generator.fasttext(dim=25)
# representations_generator.fasttext(dim=50)
# representations_generator.fasttext(dim=100)
# representations_generator.fasttext(dim=200)
# representations_generator.sent2vec(dim=25)
# representations_generator.sent2vec(dim=50)
# representations_generator.sent2vec(dim=100)
# representations_generator.sent2vec(dim=200)


# ------------------------------------------------------------------
# DATA SERIALIZATION
# ------------------------------------------------------------------
# from src.serializer import DataSerializer
# s = DataSerializer()
# s.save_generators('bow')
# s.save_generators('tf_idf', 'word')
# s.save_generators('tf_idf', 'ngram')
# s.save_generators('tf_idf', 'char')
# s.save_words('glove', 25, 50)
# s.save_words('glove', 50, 50)
# s.save_words('glove', 100, 50)
# s.save_words('glove', 200, 50)
# s.save_words('fasttext', 25, 50)
# s.save_words('fasttext', 50, 50)
# s.save_words('fasttext', 100, 50)
# s.save_words('fasttext', 200, 50)
# s.save_sentences(25)
# s.save_sentences(50)
# s.save_sentences(100)
# s.save_sentences(200)


# ------------------------------------------------------------------
# EXPERIMENTS EXECUTION
# ------------------------------------------------------------------
# from src.experiments.executor import ExperimentExecutor
# from src.models.classic import *
# ExperimentExecutor.execute(LogisticRegression)
# ExperimentExecutor.execute(SVM)
# ExperimentExecutor.execute(RandomForest)
# ExperimentExecutor.execute(NaiveBayes)
