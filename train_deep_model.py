import json

from src.serializer import *
from src.models.KerasModel import KerasModel
from sklearn.model_selection import train_test_split


with open('config.json') as fr:
    config = json.load(fr)['general']
    config['full'] = config['full'] == 'True'


# s = DataSerializer(full = config['full'])
# s.save_words(model=config['emb_method'], dim=config['emb_dim'], size=config['max_words'])

d = DataDeserializer()
embeddings, X, y, X_test = d.load_words(model=config['emb_method'], dim=config['emb_dim'], size=config['max_words'])
X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size = float(config['test_perc']), random_state = config['split_seed'])


params = {
    'units': 200,
    
    'dropout': 0.055,
    'recurrent_dropout': 0.025,
    
    'activation': 'tanh',
    'final_activation': 'sigmoid',
    
    'loss': 'binary_crossentropy',
    
    'optimizer': 'adam',
    'learning_rate': 0.00075,
    
    'metrics': ['accuracy'],
    
    'epochs': 5,
    'batch_size': 2 ** 12
}

# model = KerasModel(config['model'])
# model.init_model(embeddings, params) 
# model.train((X_tr, y_tr), (X_vl, y_vl), retrain = False)
# model.predict(X_test)

model = KerasModel(config['model'])
model.init_model(embeddings, params)
model.load(params)
# model.train((X_tr, y_tr), (X_vl, y_vl), retrain = True)
model.predict(X_test)