import json

from src.serializer import *
from src.models.KerasModel import KerasModel
from sklearn.model_selection import train_test_split


# Load the configuration of the model
with open('config.json') as fr:
    config = json.load(fr)['general']
    config['full'] = config['full'] == 'True'


# Generate embeddings if necessary. We should remove the comments if we want to do so.
# s = DataSerializer(full = config['full'])
# s.save_words(model = config['emb_method'], dim = config['emb_dim'], size = config['max_words'])

# Load the necessary embeddings. We assume the embeddings already exists. Run the previous 2 lines if we need to generate embeddings.
d = DataDeserializer()
embeddings, X, y, X_test = d.load_words(model = config['emb_method'], dim = config['emb_dim'], size = config['max_words'])

# Split the data into train and validation datasets.
X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size = float(config['test_perc']), random_state = config['split_seed'])


# Define hyperparameters of the architecture of the model.
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


# This chunk of 4 lines is used to train a new model.

# Initialize a new model whose type is specified in the configuration file.
model = KerasModel(config['model'])
# Initialize the model with the embedding matrix.
model.init_model(embeddings, params) 
# Train the model.
model.train((X_tr, y_tr), (X_vl, y_vl), retrain = False)
# Make the final predictions.
model.predict(X_test)



# This chunk of 5 lines is used to load and (retrain if necessary a model) and make final predictions.

# Initialize a new model whose type is specified in the configuration file.
model = KerasModel(config['model'])
# Initialize the model with the embedding matrix.
model.init_model(embeddings, params)
# Load the last pre-trained model (or specify path if weights are at another location).
model.load(params)
# Retrain the model if necessary.
model.train((X_tr, y_tr), (X_vl, y_vl), retrain = True)
# Make final predictions.
model.predict(X_test)