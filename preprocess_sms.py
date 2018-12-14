from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import unicodedata
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os


tbl = dict.fromkeys(
    i for i in range(sys.maxunicode)
    if unicodedata.category(chr(i)).startswith('P')
)


def load_sms_data(path):
    return pd.read_csv(path, encoding='utf-8', sep=',', names=["class", "msg"])


def remove_punctuation(text):
    return text.translate(tbl)


def clean_sms(sms_message):
    return remove_punctuation(sms_message).lower()


def class_str_to_int(class_str):
    if class_str == "ham":
        return -1
    elif class_str == "spam":
        return 1
    else:
        return None


def accuracy(preds, targets):
    return ((preds == targets).sum())/len(targets)


sms_data_df = load_sms_data("Data/sms/smsspamcollection/sms_spam3.csv")

# clean messages
sms_data_df['clean_msg'] = sms_data_df['msg'].astype('str').apply(clean_sms)
sms_data_df['target'] = sms_data_df['class'].astype(
    'str').apply(class_str_to_int)


# perform classification
x = sms_data_df['clean_msg']
y = sms_data_df['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=44)

print("Train set: %s | %s" % (len(x_train), len(y_train)))
print("Test set: %s | %s" % (len(x_test), len(y_test)))

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

train_matrix = vectorizer.fit_transform(x_train)
test_matrix = vectorizer.transform(x_test)

model = LogisticRegression(
    random_state=0, solver='lbfgs', multi_class='multinomial')
model.fit(train_matrix, y_train)
test_preds = model.predict(test_matrix)

# accurary
print("Accuracy for train set:\t%.4f" % accuracy(
    model.predict(train_matrix), y_train))
print("Accuracy for test set:\t%.4f" % accuracy(test_preds, y_test))
