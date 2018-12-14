# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from vectorizers import VectCount
from vectorizers import VectTfidf

from classifiers import classifiers

email_df = pd.read_pickle('email_cleaned')

X_train, X_test, y_train, y_test = train_test_split(email_df.clean_msg, email_df.target, test_size=0.3)

train_matrix_count, test_matrix_count, features_count = VectCount(X_train, X_test)
train_matrix_Tfidf, test_matrix_Tfidf, features_Tfidf = VectTfidf(X_train, X_test)



def accuracy(preds, targets):
    return ((preds == targets).sum())/len(targets)

def perform_model(model_obj, train_matrix, test_matrix, features):
    model = model_obj['model']
    model.fit(train_matrix, y_train)
    # print('Test score: {}'.format(model.score(test_matrix, y_test)))
    coefs_sorted, features_sorted = zip(*sorted(zip(model.coef_[0],features)))
    y_test_pred = model.predict(test_matrix)
    y_train_pred = model.predict(train_matrix)
    train_acc = accuracy(y_train_pred, y_train)
    test_acc = accuracy(y_test_pred, y_test)
    #model.predict_proba(test_matrix)
    return model, coefs_sorted, features_sorted, train_acc, test_acc
    

for classifier in classifiers:
    model_count, coefs_sorted_count, features_sorted_count, train_acc_count, test_acc_count = perform_model(classifier, train_matrix_count, test_matrix_count, features_count)
    model_Tfidf, coefs_sorted_Tfidf, features_sorted_Tfidf, train_acc_Tfidf, test_acc_Tfidf = perform_model(classifier, train_matrix_Tfidf, test_matrix_Tfidf, features_Tfidf)

    print("Model name: %s" % classifier['name'])
    print("------------------------------------")
    print("Count Vectorizer")
    print("Accuracy for: \t train set: %.3f \t test set: %.3f" % (train_acc_count, test_acc_count))
    print("Tfidf Vectorizer")
    print("Accuracy for: \t train set: %.3f \t test set: %.3f" % (train_acc_Tfidf, test_acc_Tfidf))
    print()
