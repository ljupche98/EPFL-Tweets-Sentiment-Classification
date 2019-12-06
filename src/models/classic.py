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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


MODELS_DIR = 'data/models/'
CV = 5


def print_stats(clf, data):
    X_train, X_test, y_train, y_test = data
    print('Optimal parameters:')
    print('', clf.best_params_)
    print('Score:')
    print(f' Train set: {round(clf.score(X_train, y_train) * 100, 2)}%')
    print(f'  Test set: {round(clf.score(X_test, y_test) * 100, 2)}%')


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
            'alpha': np.linspace(0.01, 1, 10)
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
            'alpha': np.linspace(0.01, 1, 10)
        }
        clf = GridSearchCV(SGDClassifier(loss='hinge',
                                        penalty='l2'),
                        parameters,
                        cv=CV)
        clf.fit(X_train, y_train)
        # persist_model(clf, mname)
    print_stats(clf, (X_train, X_test, y_train, y_test))


def RandomForest(X_train, X_test, y_train, y_test, load=False):
    mname = 'RandomForest'
    if load:
        clf = load_model(mname)
    else:
        parameters = { }
        clf = GridSearchCV(RandomForestClassifier(n_estimators=100,
                                                max_depth=2),
                        parameters,
                        cv=CV)
        clf.fit(X_train, y_train)
        # persist_model(clf, mname)
    print_stats(clf, (X_train, X_test, y_train, y_test))


def NaiveBayes(X_train, X_test, y_train, y_test, load=False):
    # TODO: Implement
    pass
