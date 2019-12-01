from src.serializer import *

# s = DataSerializer()
# s.save_generators('bow')
# s.save_generators()
# s.save_words()
# s.save_sentences()

s = DataDeserializer()
s.load_generators('bow')
s.load_generators()
s.load_words()
s.load_sentences()