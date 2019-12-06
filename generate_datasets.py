# ------------------------------------------------------------------
# DATA SERIALIZATION
# ------------------------------------------------------------------
from src.serializer import DataSerializer

if __name__ == '__main__':
    s = DataSerializer()

    s.save_generators('bow')
    s.save_generators('tf_idf', 'word')
    s.save_generators('tf_idf', 'ngram')
    s.save_generators('tf_idf', 'char')
    s.save_words('glove', 25, 50)
    s.save_words('glove', 50, 50)
    s.save_words('glove', 100, 50)
    s.save_words('glove', 200, 50)
    s.save_words('fasttext', 25, 50)
    s.save_words('fasttext', 50, 50)
    s.save_words('fasttext', 100, 50)
    s.save_words('fasttext', 200, 50)
    s.save_sentences(25)
    s.save_sentences(50)
    s.save_sentences(100)
    s.save_sentences(200)
