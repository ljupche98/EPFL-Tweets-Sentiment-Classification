import json

from src.serializer import *
from src.models.KerasModel import KerasModel
from sklearn.model_selection import train_test_split


with open('config.json') as fr:
    config = json.load(fr)['general']

# s = DataSerializer()
# s.save_words(model=config['emb_method'], dim=config['emb_dim'], size=config['max_words'])

d = DataDeserializer()
embeddings, X, y = d.load_words(model=config['emb_method'], dim=config['emb_dim'], size=config['max_words'])

X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size = 0.15, random_state = config['split_seed'])

LSTM_params = {
    'units': 100,
    'dropout': 0,
    'recurrent_dropout': 0,
    'activation': 'sigmoid',
    'loss': 'binary_crossentropy',
    'optimizer': 'RMSprop',
    'metrics': ['accuracy'],
    'epochs': 5,
    'batch_size': 512
}

test_model = KerasModel('LSTM')
test_model.init_model(embeddings, LSTM_params)
test_model.train((X_tr, y_tr), (X_vl, y_vl), retrain = False)
# test_model.predict()