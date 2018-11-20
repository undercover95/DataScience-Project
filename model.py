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
    train_matrix_count = vectorizer_count.fit(X_train)
    test_matrix_count = vectorizer_count.transform(X_test)
    return train_matrix_count, test_matrix_count

def VectTfidf(X_train, X_test):
    vectorizer_tfidf = TfidfVectorizer(token_pattern = r'\b\w+\b')
    train_matrix_tfidf = vectorizer_tfidf.fit(X_train)
    test_matrix_tfidf = vectorizer_tfidf.transform(X_test)
    return train_matrix_tfidf, test_matrix_tfidf


train_matrix_count, test_matrix_count = VectCount(X_train, X_test)

model_bayes = MultinomialNB()
model_bayes.fit(train_matrix_count, y_train)
model_bayes.predict(test_matrix_count)
model_bayes.score(test_matrix_count, y_test)


"""
model_linear = LinearRegression()
model_linear.fit(train_matrix_count, y_train)
model_linear.predict(test_matrix_count)
model_linear.score(test_matrix_count, y_test)
"""
