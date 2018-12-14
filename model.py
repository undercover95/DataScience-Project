# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:28:00 2018

@author: portable Otto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB

email_df = pd.read_pickle('email_cleaned')

X_train, X_test, y_train, y_test = train_test_split(email_df.clean_msg, email_df.target, test_size=0.3, random_state=25)

def VectCount(X_train, X_test):
    vectorizer_count = CountVectorizer(token_pattern = r'\b\w+\b')
    train_matrix_count = vectorizer_count.fit_transform(X_train)
    test_matrix_count = vectorizer_count.transform(X_test)
    return train_matrix_count, test_matrix_count

def VectTfidf(X_train, X_test):
    vectorizer_tfidf = TfidfVectorizer(token_pattern = r'\b\w+\b')
    train_matrix_tfidf = vectorizer_tfidf.fit_transform(X_train)
    test_matrix_tfidf = vectorizer_tfidf.transform(X_test)
    return train_matrix_tfidf, test_matrix_tfidf


train_matrix_count, test_matrix_count = VectCount(X_train, X_test)
train_matrix_Tfidf, test_matrix_Tfidf = VectTfidf(X_train, X_test)


def model(model_name, train_matrix, test_matrix):
    if model_name=='Bayes':
        model = MultinomialNB()
    elif model_name=='Linear':
        model = LinearRegression()
    else:
        return None
    model.fit(train_matrix, y_train)
    print('Test score: {}'.format(model.score(test_matrix, y_test)))
    return model

print('Lets start the fitting!')

model_bayes_count = model('Bayes',train_matrix_count, test_matrix_count)

print('Model number 2')

model_bayes_Tfidf = model('Bayes',train_matrix_Tfidf, test_matrix_Tfidf)

#model_linear_count = model('Linear',train_matrix_count, test_matrix_count)

#model_linear_Tfidf = model('Linear',train_matrix_Tfidf, test_matrix_Tfidf)
