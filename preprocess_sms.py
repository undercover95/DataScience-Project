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

def clean_sms_data_and_save(output_filename):
    sms_data_df = load_sms_data("Data/sms/smsspamcollection/sms_spam3.csv")
    sms_data_df['clean_msg'] = sms_data_df['msg'].astype('str').apply(clean_sms)
    sms_data_df['target'] = sms_data_df['class'].astype(
        'str').apply(class_str_to_int)


    sms_data_df.to_pickle(output_filename)