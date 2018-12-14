import glob
import tarfile
import pandas as pd
import numpy as np
import tqdm
import string
import unicodedata
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def load_data_files(path):
    ham = []
    spam = []

    files = glob.glob(path)

    for name in files:
        tar = tarfile.open(name, "r:gz")
        for member in tar.getmembers():
            if('ham' in member.name):
                f = tar.extractfile(member)
                if f is not None:
                    ham.append(f.read())
            elif('spam' in member.name):
                f = tar.extractfile(member)
                if f is not None:
                    spam.append(f.read())

    return spam, ham


def decode_emails(emails):
    subject = b"\n"
    tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                        if unicodedata.category(chr(i)).startswith('P'))

    subjects = []
    to_del = []

    for i in tqdm.tqdm(range(len(emails))):
        index = emails[i].find(subject)
        # from index 9 to get rid of 'Subject: '
        subjects.append(emails[i][9:index])
        text = emails[i][index + len(subject):]

        if text != b'':
            try:
                emails[i] = text.decode("utf-8").replace("\n",
                                                         " ").replace("\r", " ").translate(tbl).lower()
            except UnicodeDecodeError:
                to_del.append(i)
        else:
            emails[i] = ''

    for item in reversed(to_del):  # removing all emails that couldnt be decoded
        del emails[item]
        del subjects[item]

    return emails  # return decoded emails
