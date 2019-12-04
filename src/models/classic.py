''' This script contains several classical ML algorithms
for text classification it is used to conduct the experiments
for our study.

Note: If you want to add your own experiment, implement a function
with the following signature: 
    Model(X_train, X_test, y_train, y_test)
that does proper grid search over the hyperparameter space and
cross-validates the parameters using 5 folds.

TODO: Store the results in a file.
'''
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def LogisticRegression(X_train, X_test, y_train, y_test):
    parameters = {
        'alpha': np.linspace(0.01, 1, 10)
    }
    clf = GridSearchCV(RidgeClassifier(),
                       parameters,
                       cv=5)
    clf.fit(X_train, y_train)
    print('Optimal parameters:')
    print('', clf.best_params_)
    print('Score:')
    print(f' Train set: {round(clf.score(X_train, y_train) * 100, 2)}%')
    print(f'  Test set: {round(clf.score(X_test, y_test) * 100, 2)}%')


def SVM(X_train, X_test, y_train, y_test):
    parameters = {
        'alpha': np.linspace(0.01, 1, 10)
    }
    clf = GridSearchCV(SGDClassifier(loss='hinge',
                                     penalty='l2'),
                       parameters,
                       cv=5)
    clf.fit(X_train, y_train)
    print('Optimal parameters:')
    print('', clf.best_params_)
    print('Score:')
    print(f' Train set: {round(clf.score(X_train, y_train) * 100, 2)}%')
    print(f'  Test set: {round(clf.score(X_test, y_test) * 100, 2)}%')


def RandomForest(X_train, X_test, y_train, y_test):
    parameters = { }
    clf = GridSearchCV(RandomForestClassifier(n_estimators=100,
                                              max_depth=2),
                       parameters,
                       cv=5)
    clf.fit(X_train, y_train)
    print('Optimal parameters:')
    print('', clf.best_params_)
    print('Score:')
    print(f' Train set: {round(clf.score(X_train, y_train) * 100, 2)}%')
    print(f'  Test set: {round(clf.score(X_test, y_test) * 100, 2)}%')


def NaiveBayes(X_train, X_test, y_train, y_test):
    # TODO: Implement
    pass
