from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import unicodedata
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
import os
import time

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    {
        'name': "Nearest Neighbors k=3",
        'model': KNeighborsClassifier(3)
    },
    {
        'name': "Nearest Neighbors k=30",
        'model': KNeighborsClassifier(30)
    },
    {
        'name': "Nearest Neighbors k=100",
        'model': KNeighborsClassifier(100)
    },
    {
        'name': "Linear SVM",
        'model': SVC(kernel="linear", C=0.025)
    },
    {
        'name': "Decision Tree",
        'model': DecisionTreeClassifier(max_depth=5)
    },
    {
        'name': "Random Forest",
        'model': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    },
    {
        'name': "Naive Bayes",
        'model': MultinomialNB()
    },
    {
        'name': "Logistic Regression",
        'model': LogisticRegression()
    }
]

def perform_model(model_obj, train_matrix, test_matrix, features):
    model = model_obj['model']
    model.fit(train_matrix, y_train)
    print('Test score: {}'.format(model.score(test_matrix, y_test)))
    coefs_sorted, features_sorted = zip(*sorted(zip(model.coef_[0],features)))
    y_test_pred = model.predict(test_matrix)
    y_train_pred = model.predict(train_matrix)
    train_acc = accuracy(y_train_pred, y_train)
    test_acc = accuracy(y_test_pred, y_test)
    #model.predict_proba(test_matrix)
    return model, coefs_sorted, features_sorted, train_acc, test_acc