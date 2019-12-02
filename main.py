from src.serializer import *

s = DataSerializer()
# s.save_generators('bow')
# s.save_generators('tf_idf')
# s.save_words('glove')
s.save_words('fasttext')
# s.save_sentences()


# from src.tweets.model import TweetsModel
# from src.representations.generator import RepresentationsGenerator

# tweets_model = TweetsModel()
# representations_generator = RepresentationsGenerator(tweets_model)

# representations_generator.fasttext(dim=25)

