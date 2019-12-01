from src.representations.generator import RepresentationsGenerator

from src.tweets.model import TweetsModel

tweets_model = TweetsModel()
representations_generator = RepresentationsGenerator(tweets_model)

representations_generator.fasttext(dim=25)
representations_generator.fasttext(dim=50)

representations_generator.sent2vec(dim=25)
representations_generator.sent2vec(dim=50)