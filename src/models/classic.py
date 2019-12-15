''' This script contains several classical ML algorithms
for text classification it is used to conduct the experiments
for our study.

Note: If you want to add your own experiment, implement a function
with the following signature: 
    Model(X_train, X_test, y_train, y_test)
that does proper grid search over the hyperparameter space and
cross-validates the parameters using 5 folds.
'''
import numpy as np
import pickle
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

MODELS_DIR = 'data/models/'
CV = 5


def print_stats(clf, data):
    X_train, X_test, y_train, y_test = data
    print('Optimal parameters:')
    print('', clf.best_params_)
    print('Score:')
    print(f' Train set:      {round(clf.score(X_train, y_train) * 100, 2)}%')
    print(f' Validation set: {round(clf.score(X_test, y_test) * 100, 2)}%')


def persist_model(clf, mname):
    with open(MODELS_DIR + mname, 'wb') as file:
        pickle.dump(clf, file)


def load_model(mname):
    with open(MODELS_DIR + mname, 'rb') as file:
        return pickle.load(file)


def LogisticRegression(X_train, X_test, y_train, y_test, load=False):
    mname = 'LogisticRegression'
    if load:
        clf = load_model(mname)
    else:
        parameters = {
            'alpha': np.linspace(0.0001, 1, 101)
        }
        clf = GridSearchCV(RidgeClassifier(),
                           parameters,
                           cv=CV)
        clf.fit(X_train, y_train)
        # persist_model(clf, mname)
    print_stats(clf, (X_train, X_test, y_train, y_test))


def SVM(X_train, X_test, y_train, y_test, load=False):
    mname = 'SVM'
    if load:
        clf = load_model(mname)
    else:
        parameters = {
            'alpha': np.linspace(0.0001, 1, 11)
        }
        kernel = Nystroem(kernel='polynomial',
                          gamma=2,
                          n_components=400)
        kernel.fit(X_train)
        clf = GridSearchCV(SGDClassifier(),
                           parameters,
                           cv=CV)
        clf.fit(kernel.transform(X_train), y_train)
        # persist_model(clf, mname)
    print_stats(clf, (kernel.transform(X_train), kernel.transform(X_test), y_train, y_test))


def RandomForest(X_train, X_test, y_train, y_test, load=False):
    mname = 'RandomForest'
    if load:
        clf = load_model(mname)
    else:
        parameters = {
                'min_samples_leaf': [3,5,25,50]
                'n_etimators' : [25, 100, 200]
                'max_features' : ['sqrt', 'log2']
        }
        clf = GridSearchCV(RandomForestClassifier( bootstrap=True, n_jobs=-1, random_state=50),
                           parameters,
                           cv=CV)
        clf.fit(X_train, y_train)
        # persist_model(clf, mname)
    print_stats(clf, (X_train, X_test, y_train, y_test))


def NaiveBayes(X_train, X_test, y_train, y_test, load=False):
    # TODO: Implement
     mname = 'NaiveBayes'
    if load:
        clf = load_model(mname)
    else:
        if np.any(X_train<0):
            parameters = {
                'var_smoothing':[1e-5,1e-7,1e-9,1e-11,1e-13]
            }
            clf = GridSearchCV(GaussianNB(),
                               parameters,
                               cv=CV)
            clf.fit(X_train, y_train)
            # persist_model(clf, mname)
        else:
            parameters = {
                'alpha':np.linspace(0.0001,1,101)
            }
            clf = GridSearchCV(MultinomialNB(),
                               parameters,
                               cv=CV)
            clf.fit(X_train, y_train)
            # persist_model(clf, mname)
            
    print_stats(clf, (X_train, X_test, y_train, y_test))
    
    pass
