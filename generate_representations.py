# ------------------------------------------------------------------
# REPRESENTATIONS GENERATION
# ------------------------------------------------------------------
from src.tweets.model import TweetsModel
from src.representations.generator import RepresentationsGenerator

if __name__ == '__main__':
    tweets_model = TweetsModel()

    representations_generator = RepresentationsGenerator(tweets_model)
    
    representations_generator.fasttext(dim=25)
    representations_generator.fasttext(dim=50)
    representations_generator.fasttext(dim=100)
    representations_generator.fasttext(dim=200)
    representations_generator.sent2vec(dim=25)
    representations_generator.sent2vec(dim=50)
    representations_generator.sent2vec(dim=100)
    representations_generator.sent2vec(dim=200)
